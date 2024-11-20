import copy
from itertools import chain, combinations, product
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.signal import firwin, freqz
import selfeeg
from selfeeg.models import (
    ConstrainedConv2d,
    ConstrainedConv1d,
    ConstrainedDense,
    DepthwiseConv2d,
    SeparableConv2d,
)
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "HybridNetEncoder",
    "HybridNet",
    "PSDBlock",
    "PSDBlock2",
    "PSDNetEncoder",
    "PSDNet",
    "SubjectEEGModel",
    "SubjectEEGModel2",
]


# -----------------------------
#       Subject head model
# -----------------------------
class ShallowNetEncoder2(nn.Module):

    def __init__(
        self, Chans,
        F1=40, K1=25, F2=40, Pool=75, p=0.2,
        log_activation_base="e", norm_type='batchnorm',
        random_temporal_filter = True,
        spatial_depthwise = False,
        spatial_only_positive = False,
        global_pooling = False,
        Fs=-1, freeze_temporal=0,
        bias = [True, True, True],
        seed = None
    ):

        super(ShallowNetEncoder2, self).__init__()

        # Set seed before initializing layers
        self.custom_seed = seed
        if self.custom_seed is not None:
            torch.manual_seed(self.custom_seed)
            np.random.seed(self.custom_seed)
            random.seed(self.custom_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.custom_seed)
                torch.cuda.manual_seed_all(self.custom_seed)
                
        self.Fs = Fs
        self.chans = Chans
        self.freeze_temporal = freeze_temporal
        self.spatial_only_positive = spatial_only_positive
        self.bias_1conv = bias[0]
        self.bias_2conv = bias[1]
        self.bias_dense = bias[2]
        self.do_global_pooling = global_pooling
        if self.Fs <=0 and not(random_temporal_filter):
            raise ValueError(
                "to properly initialize non random temporal fir filters, "
                "Fs (sampling rate) must be given"
            )
        
        if random_temporal_filter:
            self.conv1 = nn.Conv2d(1, F1, (1, K1), stride=(1, 1), bias=self.bias_1conv)
        else:
            self.conv1 = nn.Conv2d(1, F1, (1, K1), stride=(1,1), bias=self.bias_1conv)
            self._initialize_custom_temporal_filter(self.custom_seed)

        if spatial_depthwise:
            self.conv2 = nn.Conv2d(F1, F2, (Chans, 1), stride=(1, 1), groups = F1, bias = self.bias_2conv)
        else:
            self.conv2 = nn.Conv2d(F1, F2, (Chans, 1), stride=(1, 1), bias = self.bias_2conv)
            
        if "batch" in norm_type.casefold(): 
            self.batch1 = nn.BatchNorm2d(F2,affine=True)
        elif "instance" in norm_type.casefold():
            self.batch1 = nn.InstanceNorm2d(F2)
        else:
            raise ValueError(
                "normalization layer type can be 'batchnorm' or 'instancenorm'"
            )
        
        if log_activation_base in ["e", torch.e]: 
            self.log_activation = lambda x: torch.log(torch.clamp(x, 1e-7, 1e4))
        elif log_activation_base in ["10", 10]:
            self.log_activation = lambda x: torch.log10(torch.clamp(x, 1e-7, 1e4))
        elif log_activation_base in ["db", "dB"]:
            self.log_activation = lambda x: 10*torch.log10(torch.clamp(x, 1e-7, 1e4))
        elif log_activation_base == "logrelu":
            self.log_activation = lambda x: torch.log(torch.nn.functional.relu(x)+1)
        elif log_activation_base in ["linear"]:
            self.log_activation = lambda x: x
        else:
            raise ValueError(
                "allowed activation base are 'e' for torch.log, "
                "'10' for torch.log10, and 'dB' for 10*torch.log10"
            )
        
        if not self.do_global_pooling:
            self.pool2 = nn.AvgPool2d((1, Pool), stride=(1, max(1, Pool//5)))
        else:
            self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        self.drop1 = nn.Dropout(p)
        self.flatten = nn.Flatten()

    def forward(self, x):
        if self.freeze_temporal:
            self.freeze_temporal -= 1
            self.conv1.requires_grad_(False)
        else:
            self.conv1.requires_grad_(True)        
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        if self.spatial_only_positive:
            x = self.conv2._conv_forward(x, self._get_spatial_softmax(), self.conv2.bias)
        else:
            x = self.conv2(x)

        x = self.batch1(x)
        x = torch.square(x)
        
        if self.do_global_pooling:
            x = self.global_pooling(x)
        else:
            x = self.pool2(x)

        x = self.log_activation(x)
        x = self.drop1(x)
        x = self.flatten(x)
        
        return x

    @torch.no_grad()
    def _get_spatial_softmax(self):
        return torch.softmax(self.conv2.weight, -2)

    @torch.no_grad()
    def _get_spatial_zero(self):
        return self.conv2.weight-torch.sum(self.conv2.weight,-2, keepdim=True)
    
    @torch.no_grad()
    def _initialize_custom_temporal_filter(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        if self.conv1.weight.shape[-1] >= 75:
            bands = (
                ( 0.5,  4.0), #delta
                ( 4.0,  8.0), #theta
                ( 8.0, 12.0), #alpha
                (12.0, 16.0), #beta1 
                (16.0, 20.0), #beta2
                (20.0, 28.0), #beta3
                (28.0, 45.0)  #gamma
            )
        else:
            bands = (
                ( 0.5,  8.0),
                ( 8.0, 16.0),
                (16.0, 28.0),
                (28.0, 45.0)
            )
        F, KernLength = self.conv1.weight.shape[0], self.conv1.weight.shape[-1]
        comb = self._powerset(bands)
        #if F <= len(comb):
        for i in range(np.min([F,len(comb)])):
            filt_coeff = firwin(
                KernLength,
                self._merge_tuples(comb[i]),
                pass_zero=False,
                fs=self.Fs
            )
            self.conv1.weight.data[i,0,0] = torch.from_numpy(filt_coeff)

    @torch.no_grad()
    def _powerset(self, s):
        return tuple(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))

    @torch.no_grad()
    def _merge_tuples(self, tuples):
        merged = [num for tup in tuples for num in tup]
        merged = sorted(merged)
        if len(merged)>2:
            new_merged = [merged[0]]
            for i in range(1, len(merged)-2, 2):
                if merged[i] != merged[i+1]:
                    new_merged.append(merged[i])
                    new_merged.append(merged[i+1])
            new_merged.append(merged[-1])
            return sorted(new_merged)  
        return merged

    @torch.no_grad()
    def _combinatorial_op(self, N, k):
        return int((math.factorial(N))/(math.factorial(k)*math.factorial(N-k)))

    @torch.no_grad()
    def plot_temporal_response(self, filter):
        b = self.conv1.weight.data[filter]
        b = b.detach().flatten().numpy()
        w, h = freqz(b)
        fig, ax1 = plt.subplots(figsize=(15,8))
        ax1.set_title(f'Temporal filter number {filter} frequency response', fontsize=22)
        ax1.plot(w*((self.Fs/2)/torch.pi), 20 * np.log10(abs(h)), 'b')
        ax1.set_ylabel('Amplitude [dB]', color='b', fontsize=20)
        ax1.set_xlabel('Frequency [rad/sample]', fontsize=20)
        ax1.set_xticks([i*4 for i in range(17)])
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w*((self.Fs/2)/torch.pi), angles, 'g')
        ax2.set_ylabel('Angle (radians)', color='g', fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.axis('tight')
        ax1.grid(True)
        ax2.grid(False)
        plt.show()
        
        
class ShallowNet2(nn.Module):
    def __init__(
        self, nb_classes, Chans, Samples, F1=40, K1=25, F2=40, Pool=75, p=0.2,
        log_activation_base="e", norm_type='batchnorm',
        random_temporal_filter = True,
        spatial_depthwise = False,
        spatial_only_positive = False,
        global_pooling = False, 
        bias = [True, True, True],
        Fs=-1, freeze_temporal=0, dense_hidden=None, return_logits=True, seed = None
    ):

        super(ShallowNet2, self).__init__()

        # Set seed before initializing layers
        self.custom_seed = seed
        if self.custom_seed is not None:
            torch.manual_seed(self.custom_seed)
            np.random.seed(self.custom_seed)
            random.seed(self.custom_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.custom_seed)
                torch.cuda.manual_seed_all(self.custom_seed)

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = ShallowNetEncoder2(
            Chans, F1, K1, F2, Pool, p, log_activation_base,
            norm_type,
            random_temporal_filter,
            spatial_depthwise,
            spatial_only_positive,
            global_pooling,
            Fs, freeze_temporal, bias, seed
        )
        if global_pooling:
            self.emb_size = F2
        else:
            self.emb_size = F2 * ((Samples - K1 + 1 - Pool) // max(1,int(Pool//5)) + 1)

        if self.custom_seed is not None:
            torch.manual_seed(self.custom_seed)
            np.random.seed(self.custom_seed)
            random.seed(self.custom_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.custom_seed)
                torch.cuda.manual_seed_all(self.custom_seed)
            
        if dense_hidden is None or dense_hidden<=0:
            self.Dense = nn.Linear(self.emb_size, 1 if nb_classes <= 2 else nb_classes, bias=bias[2])
        else:
            self.Dense = nn.Sequential(
                nn.Linear(self.emb_size, dense_hidden, bias=True),
                nn.ELU(alpha=1.0),
                nn.Linear(dense_hidden, 1 if nb_classes <= 2 else nb_classes,bias=bias[2])
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.Dense(x)

        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x

    @torch.no_grad()
    def plot_temporal_response(self, filter):
        self.encoder.plot_temporal_response(filter)


class CustomDenseLayer(nn.Module):
    def __init__(self, input_size, output_size, connections, initial_weights):
        super(CustomDenseLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.connections = connections
        self.weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32).view(-1, 1), requires_grad=False)  # Frozen weights
        
    def forward(self, x):
        output = torch.zeros(x.size(0), self.output_size, device=x.device)  # Initialize output tensor
        for (input_index, output_index), weight in zip(self.connections, self.weights):
            output[:, output_index] += weight.squeeze() * x[:, input_index]  # Apply weight to the input
        return output

# ------------------------------
#         ShallowNet
# ------------------------------
class ShallowNet(nn.Module):
    """
    Pytorch implementation of the ShallowNet model.

    Original paper can be found here [shall]_ .
    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    nb_classes: int
        The number of classes. If less than 2, a binary classification
        problem is considered (output dimensions will be [batch, 1] in this case).
    Chans: int
        The number of EEG channels.
    Samples: int
        The sample length. It will be used to calculate the embedding size
        (for head initialization).
    F: int, optional
        The number of output filters in the temporal convolution layer.

        Default = 8
    K1: int, optional
        The length of the temporal convolutional layer.

        Default = 25
    Pool: int, optional
        The temporal pooling kernel size.

        Default = 75
    p: float, optional
        The dropout probability. Must be in [0,1)

        Default= 0.2
    return_logits: bool, optional
        Whether to return the output as logit or probability.  It is suggested
        to not use False as the pytorch crossentropy applies the softmax internally.

        Default = True

    Note
    ----
    In this implementation, the number of channels is an argument.
    However, in the original paper authors preprocess EEG data by selecting
    a subset of only 21 channels. Since the net is very minimalist,
    please follow the authors' notes.

    References
    ----------
    .. [shall] Schirrmeister et al., Deep Learning with convolutional
      neural networks for decoding and visualization of EEG pathology,
      arXiv:1708.08012

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.ShallowNet(4,8,512)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 4])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(self, nb_classes, Chans, Samples, F=40, K1=25, Pool=75, p=0.2, return_logits=True, seed=None):

        super(ShallowNet, self).__init__()
        
        self.custom_seed = seed
        if self.custom_seed is not None:
            torch.manual_seed(self.custom_seed)
            np.random.seed(self.custom_seed)
            random.seed(self.custom_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.custom_seed)
                torch.cuda.manual_seed_all(self.custom_seed)

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = ShallowNetEncoder(Chans, F=F, K1=K1, Pool=Pool, p=p, seed=self.custom_seed)

        if self.custom_seed is not None:
            torch.manual_seed(self.custom_seed)
            np.random.seed(self.custom_seed)
            random.seed(self.custom_seed)
    
        self.Dense = nn.Linear(
            F * ((Samples - K1 + 1 - Pool) // 15 + 1), 1 if nb_classes <= 2 else nb_classes
        )

    def forward(self, x):
        """
        :meta private:
        """
        x = self.encoder(x)
        x = self.Dense(x)
        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x

class ShallowNetEncoder(nn.Module):
    """
    Pytorch implementation of the ShallowNet Encoder.

    See ShallowNet for some references.
    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    Chans: int
        The number of EEG channels.
    F: int, optional
        The number of output filters in the temporal convolution layer.

        Default = 40
    K1: int, optional
        The length of the temporal convolutional layer.

        Default = 25
    Pool: int, optional
        The temporal pooling kernel size.

        Default = 75
    p: float, optional
        Dropout probability. Must be in [0,1)

        Default= 0.2

    Note
    ----
    In this implementation, the number of channels is an argument.
    However, in the original paper authors preprocess EEG data by
    selecting a subset of only 21 channels. Since the net is very
    minimalistic, please follow the authors' notes.

    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.ShallowNetEncoder(8)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 224])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(self, Chans, F=40, K1=25, Pool=75, p=0.2, seed=None):

        super(ShallowNetEncoder, self).__init__()

        self.custom_seed = seed
        if self.custom_seed is not None:
            torch.manual_seed(self.custom_seed)
            np.random.seed(self.custom_seed)
            random.seed(self.custom_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.custom_seed)
                torch.cuda.manual_seed_all(self.custom_seed)
        
        self.conv1 = nn.Conv2d(1, F, (1, K1), stride=(1, 1))
        self.conv2 = nn.Conv2d(F, F, (Chans, 1), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(F)
        self.pool2 = nn.AvgPool2d((1, Pool), stride=(1, 15))
        self.drop1 = nn.Dropout(p)
        self.flatten2 = nn.Flatten()

    def forward(self, x):
        """
        :meta private:
        """
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch1(x)
        x = torch.square(x)
        x = self.pool2(x)
        x = torch.log(torch.clamp(x, 1e-7, 10000))
        x = self.drop1(x)
        x = self.flatten2(x)
        return x


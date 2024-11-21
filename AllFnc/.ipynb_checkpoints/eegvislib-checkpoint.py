import numpy as np
import pandas as pd
import torch
import mne
import seaborn as sns
from scipy.stats import gaussian_kde, linregress, pearsonr, norm
from scipy.signal import welch, freqz
from statsmodels.stats import multitest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import FuncFormatter

###################################
####     EXTRACT ACTIVATION     ###
###################################
def hook_fn(module, input, output, activation_store):
    """
    Hook function to capture the output activations of a specified layer.

    Parameters:
    -----------
    - module: The layer/module being hooked.
    - input: The input to the layer.
    - output: The output from the layer.
    - activation_store: List to store the output activations.
    """
    activation_store.append(output)

@torch.no_grad()
def out_activation(model, target_layer, input_data):
    """
    Captures the activations from a specified layer of the model during a forward pass.

    Parameters:
    -----------
    - model (torch.nn.Module): The neural network model.
    - target_layer (str): The dot-separated path to the layer within the model (e.g., "layer1.conv2").
    - input_data (torch.Tensor): The input tensor for the model's forward pass.

    Returns:
    --------
    - torch.Tensor: The output activations of the specified layer.
    """
    # Ensure model is in evaluation mode for consistent results
    model.eval()
    
    # List to store captured activations
    activation_store = []

    # Navigate to the specified target layer
    sub_layers = target_layer.split('.')
    target_module = model
    for layer_name in sub_layers:
        target_module = getattr(target_module, layer_name)

    # Register a hook on the target layer to capture its output
    hook_handle = target_module.register_forward_hook(
        lambda module, input, output: hook_fn(module, input, output, activation_store)
    )
    
    # Perform a forward pass with the provided input data
    model(input_data)
    
    # Remove the hook after the forward pass to avoid memory leaks or unexpected behavior
    hook_handle.remove()

    # Return the captured activation (assuming only one forward hook activation is expected)
    return activation_store[0] if activation_store else None

###################################
####     EXTRACT PSD LIMITS     ###
###################################
def get_PSDunit(norm, unit=None, power=False):
    """
    Returns the appropriate unit for Power Spectral Density (PSD) based on the normalization type.

    Parameters:
    -----------
    - norm (str): The normalization type. Options are:
        - 'log10': Returns the unit as 'dB/10'.
        - '10log10': Returns the unit as 'dB'.
        - 'linear': Returns the unit as 'muV^2/Hz'.
    - unit (str, optional): Unit of the calculated quantity, used if passed and norm is linear. Default is None.
    - power (logical, optional): If power was calculated (integral over PSD), change the unit to 'muV^2'. Default is False.

    Returns:
    --------
    - unit (str): The unit of PSD corresponding to the specified normalization type.

    Raises:
    -------
    ValueError: If norm is not one of 'log10', '10log10', or 'linear'.
    """
    
    if norm == 'log10':
        unit = '[dB/10]'
    elif norm == '10log10':
        unit = '[dB]'
    elif norm == 'linear':
        if unit is not None:
            unit=unit
        else:
            if power:
                unit = '[µV^2]'
            else:
                unit = '[µV^2/Hz]'
    else:
        raise ValueError("Invalid norm type. Expected 'log10', '10log10', or 'linear'.")

    return unit
    
def data_transform(data, norm):
    """
    Transforms data based on a specified normalization type.

    Parameters:
    -----------
    - data (float, int, np.ndarray, or torch.Tensor): 
        The input data to transform. This could be a single numeric value, a numpy array, or a PyTorch tensor.
    - norm (str): The normalization type. Options are:
        - 'log10': Applies `log10(data)`.
        - '10log10': Applies `10 * log10(data)`.
        - 'linear': Returns the data as-is without transformation.

    Returns:
    --------
    - data (float, int, np.ndarray, or torch.Tensor): 
        Transformed data based on the specified norm.

    Raises:
    -------
    ValueError: If norm is not one of 'log10', '10log10', or 'linear'.
    TypeError: If data is not a numeric value, numpy array, or torch tensor.
    """
    
    # Check if data is of a valid type
    if not isinstance(data, (int, float, np.ndarray, torch.Tensor)) and not isinstance(data, np.generic):
        raise TypeError("Data must be a numeric value, numpy array, or torch tensor.")
    
    # Apply transformations based on the normalization type
    if norm == 'log10':
        if isinstance(data, torch.Tensor):
            data = torch.log10(data)
        else:
            data = np.log10(data)
    elif norm == '10log10':
        if isinstance(data, torch.Tensor):
            data = 10 * torch.log10(data)
        else:
            data = 10 * np.log10(data)
    elif norm == 'linear':
        return data  # Return the linear data without modification
    else:
        raise ValueError("Invalid norm type. Expected 'log10', '10log10', or 'linear'.")

    return data

def get_vlim(input_tensor, sampling_rate, frequency_limits, norm):
    """
    Calculate the minimum and maximum power spectral density (PSD) values across a specific frequency range
    for each sample in the input tensor, returning these limits in dB.

    Parameters:
    -----------
    - input_tensor (torch.Tensor): The input tensor of shape (batch, channels, time) containing the signals.
    - sampling_rate (float): Sampling rate of the signals in Hz.
    - frequency_limits (tuple): Tuple (min_freq, max_freq) specifying the frequency range over which to compute PSD limits.

    Returns:
    --------
    - psd_limits (tuple): (min_psd, max_psd) indicating the global minimum and maximum PSD values 
    across the specified frequency range.
    """

    # Initialize extreme values to store the global PSD limits
    global_psd_max = -np.inf
    global_psd_min = +np.inf
    
    # Iterate over each sample in the batch
    for i in range(input_tensor.shape[0]):
        # Select the current sample from the batch
        sample_tensor = input_tensor.index_select(0, torch.tensor(i))
        
        # Compute the frequency vector and PSD values for the current sample
        freqs, psd_values = get_Pxx(torch.squeeze(sample_tensor), sampling_rate)
        
        # Identify the frequency indices within the specified range
        freq_indices = get_freq_index(freqs, frequency_limits)

        # Determine PSD limits within the selected frequency range
        if freq_indices[0] != freq_indices[1]:
            if psd_values.ndim > 1:  # Multi-channel data
                local_max_psd = np.max(psd_values[freq_indices[0]:freq_indices[1], :])
                local_min_psd = np.min(psd_values[freq_indices[0]:freq_indices[1], :])
            else:  # Single-channel data
                local_max_psd = np.max(psd_values[freq_indices[0]:freq_indices[1]])
                local_min_psd = np.min(psd_values[freq_indices[0]:freq_indices[1]])
        else:
            if psd_values.ndim > 1:  # Multi-channel data
                local_max_psd = np.max(psd_values[freq_indices[0], :])
                local_min_psd = np.min(psd_values[freq_indices[0], :])
            else:  # Single-channel data
                local_max_psd = psd_values[freq_indices[0]]
                local_min_psd = psd_values[freq_indices[0]]
        
        # Update the global PSD limits based on the local values
        global_psd_max = np.max([global_psd_max, local_max_psd])
        global_psd_min = np.min([global_psd_min, local_min_psd])
    
    # Convert the PSD limits
    psd_limits = (data_transform(global_psd_min, norm), data_transform(global_psd_max, norm))
    return psd_limits

def get_vlimscalp(full_batch_tensor, batch_tensor, sampling_rate, freq_range, num_bands, spec_dict):
    """
    Calculate the min and max power spectral density (PSD) values for scalp topography visualization,
    based on different limit strategies provided in `config`.

    Parameters:
    -----------
    - full_batch_tensor (torch.Tensor): The complete input tensor used for calculating global limits.
    - batch_tensor (torch.Tensor): A subset of `full_batch_tensor` used for calculating batch-specific limits.
    - sampling_rate (float): The sampling rate of the input data in Hz.
    - freq_range (tuple): The overall frequency range (min_freq, max_freq) for PSD calculation.
    - num_bands (int): The number of frequency bands specified in `config`.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.

    Returns:
    --------
    - psd_limits (np.ndarray): Array of shape (num_bands, 2) where each row contains the min and max PSD values (in dB)
    for a band.
    
    Notes:
    -----
    - Setting 'band' in spec_dict['vlim'] adjusts the colorbar’s min-max range individually for each band, 
    based on values from the entire full_batch_tensor.
    - Setting 'all' in spec_dict['vlim'] applies a single, consistent min-max range to the colorbar across all
    bands, based on values from the entire full_batch_tensor.
    Setting 'batchID' in spec_dict['vlim'] applies a single min-max range to the colorbar across all bands,
    but based only on the values from batch_tensor.
    """

    # Initialize array to store min and max PSD limits per band
    psd_limits = np.zeros((num_bands, 2))
    
    # Get band keys for frequency band lookup
    band_keys = list(spec_dict['bands'].keys())
    
    # Determine PSD limits based on the specified strategy
    if spec_dict['vlim'] == 'band':
        # Compute individual limits for each specified band across the full tensor
        for band_idx in range(num_bands):
            freq_range = spec_dict['bands'][band_keys[band_idx]]
            psd_limits[band_idx, :] = get_vlim(full_batch_tensor, sampling_rate, freq_range, spec_dict['lognorm'])

    elif spec_dict['vlim'] == 'all':
        # Calculate global PSD limits across the entire frequency range for all bands
        global_limits    = get_vlim(full_batch_tensor, sampling_rate, freq_range, spec_dict['lognorm'])
        psd_limits[:, 0] = global_limits[0] * np.ones(num_bands)
        psd_limits[:, 1] = global_limits[1] * np.ones(num_bands)

    elif spec_dict['vlim'] == 'batchID':
        # Calculate PSD limits for the specific batch tensor across the entire frequency range
        batch_limits     = get_vlim(batch_tensor, sampling_rate, freq_range, spec_dict['lognorm'])
        psd_limits[:, 0] = batch_limits[0] * np.ones(num_bands)
        psd_limits[:, 1] = batch_limits[1] * np.ones(num_bands)

    else: 
        raise("Colorbar normalization option, not implemented yet.")
        
    return psd_limits

def get_extended_lims(lims, percentage=5):
    """
    Extends the limits symmetrically by a specified percentage.

    Parameters:
    -----------
    lims : list or array-like
        A list or array containing two values [min, max], representing the initial limits.
    percentage : float, optional, default=5
        The percentage by which to extend the limits symmetrically. A positive value increases 
        the range, while a negative value decreases it.

    Returns:
    --------
    list
        A list containing the extended limits [new_min, new_max].

    Example:
    --------
    >>> extended_lims([0, 10], percentage=10)
    [-1.0, 11.0]
    
    This example extends the range [0, 10] by 10%, resulting in [-1.0, 11.0].
    """
    factor = 1 + percentage / 100
    return [lims[0] * factor - lims[1] * (factor - 1), 
            lims[1] * factor - lims[0] * (factor - 1)]

###################################
####     EXTRACT PSD VALUES     ###
###################################
def get_Pxx(in_wind, srate, factor=2):
    """
    Computes the Power Spectral Density (PSD) of a signal using the Welch method.

    This function calculates the PSD of the input signal using a Hanning window 
    with a default overlap of 50%. The segment length is determined based on the 
    length of the input signal and the provided `factor` parameter.

    Parameters:
    -----------
    - in_wind (torch.Tensor or numpy.ndarray): The input signal data, typically with shape (channels, samples).
    - srate (float): The sampling rate of the signal in Hz.
    - factor (int, optional): Factor to determine the segment length (default is 2). 
      The segment length is set to `len(in_wind) / factor`.

    Returns:
    --------
    - f (numpy.ndarray): The frequencies corresponding to the PSD estimate.
    - Pxx (numpy.ndarray): The estimated power spectral density of the input signal.

    Notes:
    ------
    - The function applies the Welch method with a Hanning window and a 50% overlap by default.
    - The result is returned with a small epsilon added to avoid zero values in `Pxx`.
    """
    # Ensure input is in numpy format
    in_wind = in_wind.numpy() if hasattr(in_wind, 'numpy') else in_wind
    
    # Compute segment length
    len_IN = in_wind.shape[-1]
    nperseg = len_IN // factor if len_IN % factor == 0 else np.floor(len_IN / factor).astype(int)
    
    # Compute Power Spectral Density using Welch method
    f, Pxx = welch(in_wind.T, fs=srate, nperseg=nperseg, axis=0)
    
    # Return PSD with a small epsilon to avoid division by zero
    return f, Pxx + np.finfo(float).eps

###################################
####    FREQUENCY UTILS FUNC    ###
###################################
def get_freq_index(f, f_desired):
    """
    Find the indices in the frequency array `f` that are closest to each value in `f_desired`.

    Parameters:
    -----------
    - f (np.ndarray): Array of frequencies.
    - f_desired (list of floats): Desired frequency values. Use -1 for values where no match is needed.

    Returns:
    --------
    - f_index (list of ints): List of indices corresponding to each value in `f_desired`.
    If `f_desired[i]` is -1, the corresponding index will be -1.
    """
    f_index = []
    for desired_freq in f_desired:
        if desired_freq == -1:
            index = -1  # No match requested for this frequency
        else:
            # Find the index of the closest value in `f` to `desired_freq`
            index = int(np.argmin(np.abs(f - desired_freq)))
        f_index.append(index)
    return f_index

def get_freq_names(f_des, bands):
    """
    Map each frequency in `f_des` to a band name based on the ranges specified in `bands`.

    Parameters:
    -----------
    - f_des (list of floats): List of frequencies to classify by band.
    - bands (dict): Dictionary with band names as keys and (min_freq, max_freq) tuples as values.

    Returns:
    --------
    - band_dict (dict): Dictionary where each key is an index of `f_des`, 
    and the value is the name of the band that frequency falls into.
    """
    band_dict = {}
    band_keys = list(bands.keys())
    
    for i, freq in enumerate(f_des):
        for j, key in enumerate(band_keys):
            min_freq, max_freq = bands[key]
            
            # Check frequency within the band's range, with the first band inclusive of its min_freq
            if (freq >= min_freq and freq <= max_freq) if j == 0 else (freq > min_freq and freq <= max_freq):
                band_dict[i] = key
                break  # Stop once the band is found for the current frequency
            
    return band_dict

###################################
####     CHANNELS UTILS FUNC    ###
###################################
def get_chandict(list_channels):
    """
    Create a dictionary that maps channel names to their respective indices in `list_channels`.

    Parameters:
    -----------
    - list_channels (list of str): List of channel names.

    Returns:
    --------
    - ch_dict (dict): Dictionary where the key is the channel name (str), 
    and the value is the corresponding index (int). 
    Additionally, the key 'all' will map to an array of all channel indices.
    """
    # Create a dictionary mapping each channel name to its index
    ch_dict = {channel: i for i, channel in enumerate(list_channels)}
    # Add an entry for 'all' that contains all indices
    ch_dict['all'] = np.arange(len(list_channels))
    
    return ch_dict

def get_spec_dict(keys, values):
    """
    Create a dictionary from two lists, `keys` and `values`, by pairing each element in `keys` 
    with the corresponding element in `values`. If both `keys` and `values` are empty lists, 
    an empty dictionary is returned.

    Parameters:
    -----------
    - keys (list of strings): List of keys for the dictionary.
    - values (list): List of values to be mapped to the respective keys.

    Returns:
    --------
    - spec_dict (dict): Dictionary where each key in `keys` maps to its corresponding value in `values`.
    Returns an empty dictionary if both `keys` and `values` are empty lists.
    """
    
    # Check for empty input lists
    if not keys and not values:
        return {}

    # Ensure keys and values are the same length to avoid incomplete pairings
    if len(keys) != len(values):
        raise ValueError("The length of 'keys' and 'values' must be the same.")
    
    # Create the dictionary by pairing keys with values
    spec_dict = dict(zip(keys, values))
    
    return spec_dict

###################################
####    FUNCTIONS FOR INPUT     ###
###################################
def scalp_plot(input_window, 
               spec_dict,
               inout_labels,
               batch_ID):
    """
    Plot the 1-band scalp Power Spectral Density (PSD). This function plots the PSD for each frequency band
    specified in `spec_dict['f_des']` for the given `input_window`.

    Parameters:
    -----------
    - input_window (torch.Tensor): The input tensor containing the data,
    with dimensions [batch, 1, channels, timesteps].
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - inout_labels (list of lists): A list containing true and predicted labels for all the samples in input_window.
    - batch_ID (int): Index of the batch to analyze.

    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    """
    
    # Ensure that batch_ID is a scalar if it's provided as a list
    batch_ID = batch_ID[0] if isinstance(batch_ID, list) else batch_ID
    
    # Ensure f_des is a list (it can be a single value or a list of frequencies)
    if not isinstance(spec_dict['f_des'], list):
        spec_dict['f_des'] = [spec_dict['f_des']] 
    num_freqs = len(spec_dict['f_des'])
    
    # Create a figure and subplots
    fig, ax = plt.subplots(1, 2 * num_freqs, 
                           figsize=(spec_dict['figdim'][0] * num_freqs, spec_dict['figdim'][1]), 
                           gridspec_kw={'width_ratios': [0.9, 0.1] * num_freqs},
                           constrained_layout=True)
    
    # Set the main title for the figure
    fig.suptitle(f'Batch Index: {batch_ID}| {inout_labels[0][batch_ID]} -> {inout_labels[1][batch_ID]}',
                 fontsize=spec_dict['font'] + 2)
    
    # Create MNE info and montage for plotting
    info    = mne.create_info(ch_names=spec_dict['channels'], sfreq=spec_dict['srate'], ch_types='eeg')
    montage = mne.channels.make_standard_montage(spec_dict['montage'])
    
    # Extract the data for the specified batch index
    input_window_batch = input_window.index_select(0, torch.tensor(batch_ID))
    f, Pxx    = get_Pxx(torch.squeeze(input_window_batch), spec_dict['srate'])
    f_index   = get_freq_index(f, spec_dict['f_des'])
    band_dict = get_freq_names(spec_dict['f_des'], spec_dict['bands'])  

    # Get the frequency limits for scaling the plots
    lims = get_vlimscalp(input_window, input_window_batch, 
                         spec_dict['srate'], spec_dict['flim'], 
                         num_freqs, spec_dict)

    # Loop through each frequency band to plot its topographic map and PSD
    for j in range(num_freqs):
        # Extract the PSD data for the current frequency band
        data_vector = data_transform(Pxx[f_index[j], :],spec_dict['lognorm']).T
        evoked_data = mne.EvokedArray(data_vector[:, np.newaxis], info)
        evoked_data.set_montage(montage)
    
        # Plot the topographic map and the colorbar for the current frequency band
        evoked_data.plot_topomap(show=False, axes=(ax[2 * j], ax[2 * j + 1]), show_names=True,
                                 scalings=1, vlim=extended_lims(lims[j,:]), cmap=spec_dict['cmap'],
                                 contours=spec_dict['contours'])

        # Set up the colorbar
        ax[2 * j + 1].set_yticks(np.linspace(lims[j, 0], lims[j, 1], spec_dict['cbartick']))
        ax[2 * j + 1].tick_params(labelsize=spec_dict['font'] - 4)
        unit = get_PSDunit(spec_dict['lognorm'], unit=None, power=False)
        ax[2 * j + 1].set_title(f'PSD$_{{{unit}}}$', fontsize=spec_dict['font'] - 4)

        # Set up the topography
        ax[2 * j].set_title(f'{band_dict[j]} @{f[f_index[j]]:.1f}Hz', fontsize=spec_dict['font'])
        
        for text in ax[2 * j].texts:
            text.set_fontsize(spec_dict['font'] - 8)
            text.set_fontweight('bold')
            
    return fig

###################################
####   ACTIVATION FOR 1 LAYER   ###
###################################

#### PLOT THE FM 1 LAYER IN THE TIME DOMAIN ####
def temporalFM_plot(ax, f_IN, Pxx_IN, f_FM, Pxx_FM, ch_dict, chan, FM_string, spec_dict):
    """
    Plot the temporal frequency-modulated (FM) power spectral density (PSD) for specified channels.
    Compares the PSD of the input and FM signals for the specified channels.

    Parameters:
    -----------
    - ax (matplotlib.axes.Axes): The axes object where the plot will be drawn.
    - f_IN (numpy.ndarray): Frequency values for the input signal PSD.
    - Pxx_IN (numpy.ndarray): PSD of the input signal, shape (frequencies x channels).
    - f_FM (numpy.ndarray): Frequency values for the FM signal PSD.
    - Pxx_FM (numpy.ndarray): PSD of the FM signal, shape (frequencies x channels).
    - ch_dict (dict): Dictionary mapping channel names to indices in the PSD matrices.
    - chan (list): List of channel names to plot.
    - FM_string (str): Label for the FM signal (e.g., feature map ID).
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.

    Returns:
    --------
    - ax (matplotlib.axes.Axes): The axes containing the plot.
    """
    
    # Create title string with all channel names
    s = ','.join(chan)
    
    # Initialize minimum value across all channels for labeling
    mininv = np.inf
    
    # Plot PSD for each specified channel, updating minimum PSD value
    for ch in chan:
        ax.plot(f_IN, Pxx_IN[:, ch_dict[ch]], linewidth=spec_dict['linew'], label=f'Input {ch}')
        ax.plot(f_FM, Pxx_FM[:, ch_dict[ch]], linewidth=spec_dict['linew'], label=f'$FM_{{{FM_string}}}$ {ch}')
        mininv = np.min([mininv, np.min(Pxx_IN[:, ch_dict[ch]]), np.min(Pxx_FM[:, ch_dict[ch]])])

    # Add frequency band boundaries and labels
    list_ticks = []
    for k, (f_min, f_max) in spec_dict['bands'].items():
        ax.axvline(f_max, linestyle='-', linewidth=spec_dict['linew'] / 2, color='k')
        ax.text(0.9*(f_min+f_max)/2, 0.98 * mininv, s=k, fontsize=spec_dict['font'] - 6)
        list_ticks.extend([f_min, f_max])

    # Customize ticks and labels
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=spec_dict['font'] - 4)
    ax.set_xticks(sorted(set(list_ticks)))
    ax.set_title(f'$FM_{{{FM_string}}}$ | {s}', fontsize=spec_dict['font'])
    ax.set_xlabel('$f_{[Hz]}$', fontsize=spec_dict['font'] - 2)

    unit = get_PSDunit(spec_dict['lognorm'], unit=None, power=False)
    ax.set_ylabel(f'PSD$_{{{unit}}}$', fontsize=spec_dict['font'] - 2)
    ax.legend(fontsize=spec_dict['font'] - 6, loc=spec_dict['loc'], bbox_to_anchor=(1.4, 1), borderaxespad=0.)
    ax.grid('both')
    return ax

#### PLOT THE FM 1 LAYER IN THE FREQUENCY DOMAIN (ELECTRODES) ####
def temporalFM_PSD(input_window, 
                   Mdl, layer,  
                   spec_dict, inout_labels,
                   chan_ID, batch_ID, FM_ID=None):
    """
    Plot the Power Spectral Density (PSD) of input vs feature map (FM) from a model layer.
    This function compares the PSD of the input and feature map signals for specified channels.

    Parameters:
    -----------
    - input_window (torch.Tensor): The input tensor containing the data,
    with dimensions [batch, 1, channels, timesteps].
    - Mdl (torch.nn.Module): The model to apply to the input window for generating convoluted output. 
    with keys corresponding to layer names.
    - layer (str): The name of the layer for which to plot the weights.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - inout_labels (list of lists): A list containing true and predicted labels for all the samples in input_window.
    - chan_ID (list): List of channel names to plot.
    - batch_ID (int): Index of the batch to analyze.
    - FM_ID (int or list, optional): IDs of feature maps to plot. Defaults to None (all FMs).

    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    """

    # Ensure that batch_ID is a scalar if it's provided as a list
    batch_ID = batch_ID[0] if isinstance(batch_ID, list) else batch_ID
    
    # Prepare input data on CPU and select the batch data
    input_window = input_window
    input_window_batch = torch.squeeze(input_window[batch_ID])
    f_IN, Pxx_IN = get_Pxx(input_window_batch, spec_dict['srate'])
    f_indexIN = get_freq_index(f_IN, spec_dict['flim'])
    f_IN = f_IN[f_indexIN[0]:f_indexIN[1]]
    Pxx_IN = data_transform(Pxx_IN[f_indexIN[0]:f_indexIN[1]],spec_dict['lognorm'])

    # Retrieve model's output for the selected layer and move to CPU
    convoluted_output = out_activation(Mdl, layer, input_window)
    FM_indices = FM_ID if isinstance(FM_ID, list) else list(range(convoluted_output.shape[1]))

    # Prepare channel dictionary and figure
    ch_dict = get_chandict(spec_dict['channels'])
    fig, ax = plt.subplots(len(FM_indices), 1, 
                           figsize=(spec_dict['figdim'][0], spec_dict['figdim'][1] * len(FM_indices)),
                           constrained_layout=True)
    fig.suptitle(f'Batch Index: {batch_ID} | {inout_labels[0][batch_ID]} -> {inout_labels[1][batch_ID]}',
                 fontsize=spec_dict['font'] + 2)

    # Plot each FM
    for i, fm_id in enumerate(FM_indices):
        FM = convoluted_output[batch_ID, fm_id]
        FM_string = str(fm_id)

        # Extract and log-scale PSD for FM
        f_FM, Pxx_FM = get_Pxx(torch.squeeze(FM), spec_dict['srate'])
        f_indexFM = get_freq_index(f_FM, spec_dict['flim'])
        f_FM = f_FM[f_indexFM[0]:f_indexFM[1]]
        Pxx_FM = data_transform(Pxx_FM[f_indexFM[0]:f_indexFM[1]],spec_dict['lognorm'])

        # Plot on corresponding subplot
        ax_to_plot = ax[i] if len(FM_indices) > 1 else ax
        ax_to_plot = temporalFM_plot(ax_to_plot, f_IN, Pxx_IN, f_FM, Pxx_FM, ch_dict, chan_ID, FM_string, spec_dict)
        ax_to_plot.set_xlim(spec_dict['flim'][0], spec_dict['flim'][1])

    return fig

#### PLOT THE FM 1 LAYER IN THE FREQUENCY DOMAIN (SCALP) ####
def temporalFM_PSDv2(input_window, 
                     Mdl, layer, 
                     spec_dict, inout_labels,
                     batch_ID, FM_ID=None):
    '''
    Plots the Power Spectral Density (PSD) of input versus the feature map (FM) extracted from a given model layer.
    This function visualizes PSD on scalp topographies for the specified frequency bands.

    Parameters:
    -----------
    - input_window (torch.Tensor): The input tensor containing the data,
    with dimensions [batch, 1, channels, timesteps].
    - Mdl (torch.nn.Module): The model to apply to the input window for generating convoluted output.
    - layer (str): The name of the layer for which to plot the weights.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - inout_labels (list of lists): A list containing true and predicted labels for all the samples in input_window.
    - batch_ID (int): Index of the batch to analyze.
    - FM_ID (int or list, optional): IDs of feature maps to plot. Defaults to None (all FMs).

    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    '''

    # Ensure that batch_ID is a scalar if it's provided as a list
    batch_ID = batch_ID[0] if isinstance(batch_ID, list) else batch_ID
    
    # Get model activation output from the specified layer for the given input
    convoluted_output = out_activation(Mdl, layer, input_window)
    batch, num_fms, channels, time_steps = convoluted_output.shape

    # Select the batch slice to visualize, check FM_ID to process either specific indices or all FMs
    convoluted_output_batch = convoluted_output.index_select(0, torch.tensor(batch_ID))
    FM_indices = range(num_fms) if FM_ID is None else (FM_ID if isinstance(FM_ID, list) else [FM_ID])

    # Initialize figure and axes for plotting
    num_freqs = len(spec_dict['bands'])
    fig, ax = plt.subplots(len(FM_indices), 2 * num_freqs,
                           figsize=(spec_dict['figdim'][0]*num_freqs, spec_dict['figdim'][1]*len(FM_indices)),
                           gridspec_kw={'width_ratios': [0.9, 0.1] * num_freqs},
                           constrained_layout=True)

    # Set plot title with batch and label information
    plt.suptitle(f'Batch Index: {batch_ID} | {inout_labels[0][batch_ID]} -> {inout_labels[1][batch_ID]}',
                 fontsize=spec_dict['font'] + 2)

    # Configure MNE information for plotting topographies
    info = mne.create_info(ch_names=spec_dict['channels'], sfreq=spec_dict['srate'], ch_types='eeg')
    montage = mne.channels.make_standard_montage(spec_dict['montage'])

    # Calculate limits for the color scale in topographic maps across FMs and bands
    lims = get_vlimscalp(convoluted_output, convoluted_output_batch, 
                         spec_dict['srate'], spec_dict['flim'], num_freqs, spec_dict)
        
    # Iterate over each selected feature map
    for i, fm_id in enumerate(FM_indices): 
        # Select the specific feature map to plot
        FM = convoluted_output.index_select(0, torch.tensor(batch_ID)).index_select(1, torch.tensor(fm_id))
        f_FM, Pxx_FM = get_Pxx(torch.squeeze(FM), spec_dict['srate'])

        # Iterate over specified frequency bands
        for j, (band_name, band_range) in enumerate(spec_dict['bands'].items()):
            # Get frequency indices within the current band
            f_index = get_freq_index(f_FM, band_range)

            # Compute average PSD within the band range and convert to dB
            area = np.trapz(Pxx_FM[f_index[0]:f_index[1], :].T,dx=(f_FM[1]-f_FM[0]))
            data_vector = data_transform(area/(f_FM[f_index[1]]-f_FM[f_index[0]]),spec_dict['lognorm'])

            # Create an MNE Evoked object for plotting the topographic map
            evoked_data = mne.EvokedArray(data_vector[:, np.newaxis], info)
            evoked_data.set_montage(montage)

            # Configure axis for topomap and colorbar
            ax_current = (ax[i, 2 * j], ax[i, 2 * j + 1]) if len(FM_indices) > 1 else (ax[2 * j], ax[2 * j + 1])
            
            # Plot topomap of the PSD data on the scalp
            evoked_data.plot_topomap(show=False, axes=ax_current, show_names=True,
                                     cmap=spec_dict['cmap'], scalings=1, vlim=get_extended_lims(lims[j,:]),
                                     contours=spec_dict['contours'])
            
            # Set up colorbar
            ax_current[1].tick_params(labelsize=spec_dict['font'] - 6)
            ax_current[1].set_yticks(np.linspace(lims[j, 0], lims[j, 1], spec_dict['cbartick']))
            unit = get_PSDunit(spec_dict['lognorm'],unit=None,power=True)
            ax_current[1].set_title(f'PSD$_{{{unit}}}$', fontsize=spec_dict['font'] - 4) #TO DO (UNIT BASED ON LOGNORM)

            # Set up topography
            ax_current[0].set_xticks([])
            ax_current[0].set_title(f'FM$_{{{fm_id}}}$: {band_name}', fontsize=spec_dict['font'])
            
            for text in ax_current[0].texts:
                text.set_fontsize(spec_dict['font'] - 8)
                text.set_fontweight('bold')

    return fig

###################################
####    WEIGHTS FOR 1 LAYER     ###
###################################

#### BODE's PLOT ####
def BP_plot(ax, i, h, angles, f_hz, spec_dict, string, maxminv, ind):
    """
    Plots the Bode Plot (magnitude and phase) for a bandpass filter.

    Parameters:
    -----------
    - ax (list): Array of axes to draw the plots on.
    - i (int): Index specifying the subplot in `ax` where the current filter plot is placed.
    - h (np.array): Magnitude response of the filter in linear scale.
    - angles (np.array): Phase response of the filter in radians.
    - f_hz (np.array): Frequency vector in Hz.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - string (str): Identifier string for the filter.
    - maxminv (list): Limits for magnitude and phase plots as 
    [min_magnitude, max_magnitude, min_phase, max_phase].
    - ind (int): Indicator for before (1) or after (0) training, affecting line color and style.

    Returns:
    --------
    - ax (matplotlib.axes.Axes): The axes containing the plot.
    """
    
    # Define label based on training state (because a plot without a story is just a chart!)
    label = 'After Train ' if ind == 0 else 'Before Train '
    unit = get_PSDunit(spec_dict['lognorm'], unit='[]', power=False)

    # Define frequency band tick positions and vertical markers
    list_ticks = []
    for band_name, band_limits in spec_dict['bands'].items():
        ax[i].axvline(band_limits[1], linestyle='-', linewidth=spec_dict['linew'] / 2, color='k')
        ax[i].text(0.9 * np.mean(band_limits), 0.98 * maxminv[0], s=band_name, fontsize=spec_dict['font'] - 6)
        list_ticks.append(band_limits[0])
    list_ticks.append(spec_dict['bands'][list(spec_dict['bands'].keys())[-1]][1])  # Final frequency edge

    # Plot magnitude response if specified
    if spec_dict['bodeplot'] in ['mag', 'both']:
        ax[i].plot(f_hz, data_transform(np.abs(h) + np.finfo(float).eps, spec_dict['lognorm']), color=spec_dict['colors'][ind][0],
                   linewidth=spec_dict['linew'], linestyle=spec_dict['linestyle'][ind])
        title = f"Filter$_{{{string}}}$"
        ax[i].set_title(
            title if spec_dict['filternames'] is None else f"{title}: {spec_dict['filternames'][i]}",
            fontsize=spec_dict['font']
        )

        # Configure magnitude axis labels and ticks
        ax[i].set_ylabel(f'Magnitude$_{{{unit}}}$', fontsize=spec_dict['font'] - 2, color=spec_dict['colors'][0][0])
        ax[i].set_xlabel('$f_{[Hz]}$', fontsize=spec_dict['font'] - 2)
        ax[i].tick_params(axis='both', labelsize=spec_dict['font'] - 4)
        ax[i].tick_params(axis='y', color='k', labelcolor='k')
        ax[i].tick_params(axis='x', rotation=spec_dict['rotation'])
        ax[i].set_ylim((maxminv[0], 1.3 * maxminv[1]))  # Set y-axis limits for magnitude
        ax[i].set_axisbelow(True)  # Place grid below plot elements
        ax[i].set_xticks(list_ticks)

    # Determine secondary axis for phase plot based on plotting type
    if spec_dict['bodeplot'] == 'both':
        ax2 = ax[i].twinx()
    elif spec_dict['bodeplot'] == 'phase':
        ax2 = ax[i]
    else:
        ax2 = None

    # Plot phase response if required
    if ax2 is not None:
        ax2.plot(f_hz, angles, color=spec_dict['colors'][ind][1], 
                 linewidth=spec_dict['linew'], linestyle=spec_dict['linestyle'][ind])
        
        # Set titles, labels, and ticks for phase plot
        title = f"Filter$_{{{string}}}$"
        ax2.set_title(
            title if spec_dict['filternames'] is None else f"{title}: {spec_dict['filternames'][i]}",
            fontsize=spec_dict['font'] - 2
        )
        ax2.set_ylabel('Phase$_{[rad]}$', fontsize=spec_dict['font'] - 2, color=spec_dict['colors'][0][1])
        ax2.tick_params(axis='both', labelsize=spec_dict['font'] - 6)
        ax2.tick_params(axis='y', color='k', labelcolor='k')
        ax2.tick_params(axis='x', rotation=spec_dict['rotation'])
        ax2.set_ylim((maxminv[2], 1.3 * maxminv[3]))  # Set y-axis limits for phase
        ax2.set_axisbelow(True)
        ax2.set_xticks(list_ticks)

    return ax

def BP_coefficients(kernel_weights, srate, flim):
    """
    Calculates the bandpass response from kernel weights, returning frequency responses.
    
    Parameters:
    -----------
    - kernel_weights (np.array): Weights of the kernel.
    - srate (float): Sampling rate in Hz.
    - flim (tuple): Frequency limit range as (f_min, f_max).
    
    Returns:
    --------
    - h (np.array): Magnitude response within the frequency range.
    - angles (np.array): Phase response within the frequency range.
    - f_hz (np.array): Frequency vector in Hz within the range `flim`.
    """

    w, h    = freqz(np.flip(kernel_weights.numpy()), [1])  # Frequency response
    angles  = np.unwrap(np.angle(h))  # Phase response
    f_hz    = w * srate / (2 * np.pi)  # Convert to Hz
    f_index = get_freq_index(f_hz, flim)  # Frequency range indices
    
    return h[f_index[0]:f_index[1]], angles[f_index[0]:f_index[1]], f_hz[f_index[0]:f_index[1]]

def BP_temporalkernels(Mdl_weights, layer, 
                       spec_dict,
                       filter_ID=None, kernel_ID=0, kernel_height=0, 
                       MdlBase_weights=None):
    """
    Plots Bode Plot responses of 1D temporal kernels for a specified layer in a model. 
    Each filter can optionally be compared to a baseline (pre-training) version.
    
    Parameters:
    -----------
    - Mdl_weights (dict): A dictionary containing the model's weights, 
    with keys corresponding to layer names.
    - layer (str): The name of the layer for which to plot the weights.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - filter_ID (int or list, optional): ID(s) of the filters to plot. Defaults to None (all filters).
    - kernel_ID (int, optional): ID of the kernel within the filter selected (default is 0).
    - kernel_height (int, optional): Height index within the kernel to select the relevant slice (default is 0).
    - MdlBase_weights (dict, optional): Dictionary of baseline (pre-training) weights for comparison, 
    if provided.
    
    Returns:
    --------
    - fig (matplotlib.figure.Figure): Figure containing the Bode plots for selected filters.
    - ax (matplotlib.axes.Axes): The axes containing the plot.
    """

    # Load weights for specified layer and set up baseline weights if available
    weights = Mdl_weights[layer + '.weight']
    base_weights = MdlBase_weights[layer + '.weight'] if MdlBase_weights is not None else None
    
    # Determine filter indices to plot
    if filter_ID is None:
        filters, fm, height, width = weights.shape
        filter_indices = range(filters)
    else:
        filter_indices = filter_ID if isinstance(filter_ID, list) else [filter_ID]
        filters = len(filter_indices)

    # Calculate figure layout dimensions
    nfig = int(np.ceil(np.sqrt(filters)))
    fig, ax = plt.subplots(nfig-1, nfig+1, 
                           figsize=(spec_dict['figdim'][0]*(nfig+1), spec_dict['figdim'][1]*(nfig-1)),
                           constrained_layout=True)
    fig.suptitle(spec_dict['title'], fontsize=spec_dict['font']+2)
    
    # Define legend elements for "before" and "after" training
    legend1_elements = [Line2D([0], [0], color='k', label='After Train',  linestyle=spec_dict['linestyle'][0]),
                        Line2D([0], [0], color='k', label='Before Train', linestyle=spec_dict['linestyle'][1])]
    
    ax = np.array([ax]) if nfig == 1 else ax.flatten()

    # Initialize min/max values for magnitude and phase
    min_h, max_h = np.inf, -np.inf
    min_a, max_a = np.inf, -np.inf

    # Pre-scan each filter to find global magnitude and phase limits
    for f_id in filter_indices:
        kernel_weights = weights.index_select(0, torch.tensor(f_id)).index_select(1, torch.tensor(kernel_ID)).index_select(2, torch.tensor(kernel_height))
        kernel_weights = np.squeeze(kernel_weights)
        h, angles, f_hz = BP_coefficients(kernel_weights, spec_dict['srate'], spec_dict['flim'])
        
        # Update global min/max values for magnitude and phase
        min_h = np.min([min_h, np.min(data_transform(np.abs(h) + np.finfo(float).eps,spec_dict['lognorm']))])
        max_h = np.max([max_h, np.max(data_transform(np.abs(h) + np.finfo(float).eps,spec_dict['lognorm']))])
        min_a = np.min([min_a, np.min(angles)])
        max_a = np.max([max_a, np.max(angles)])
    
    # Repeat baseline check if baseline weights are provided
    if base_weights is not None:
        for f_id in filter_indices:
            basekernel_weights = base_weights.index_select(0, torch.tensor(f_id)).index_select(1, torch.tensor(kernel_ID)).index_select(2, torch.tensor(kernel_height))
            basekernel_weights = np.squeeze(basekernel_weights)
            h, angles, f_hz = BP_coefficients(basekernel_weights, spec_dict['srate'], spec_dict['flim'])

            min_h = np.min([min_h, np.min(data_transform(np.abs(h) + np.finfo(float).eps,spec_dict['lognorm']))])
            max_h = np.max([max_h, np.max(data_transform(np.abs(h) + np.finfo(float).eps,spec_dict['lognorm']))])
            min_a = np.min([min_a, np.min(angles)])
            max_a = np.max([max_a, np.max(angles)])

    # Plot each filter’s Bode plot with the computed global min/max limits
    for i, f_id in enumerate(filter_indices):
        kernel_weights = weights.index_select(0, torch.tensor(f_id)).index_select(1, torch.tensor(kernel_ID)).index_select(2, torch.tensor(kernel_height))
        kernel_weights = np.squeeze(kernel_weights)
        h, angles, f_hz = BP_coefficients(kernel_weights, spec_dict['srate'], spec_dict['flim'])
        
        # Plot for trained weights
        ax = BP_plot(ax, i, h, angles, f_hz, spec_dict, str(f_id), [min_h, max_h, min_a, max_a], 0)
        ax[i].set_xlim(spec_dict['flim'][0], spec_dict['flim'][1])
                       
        # Plot for baseline weights if available
        if base_weights is not None:
            basekernel_weights = base_weights.index_select(0, torch.tensor(f_id)).index_select(1, torch.tensor(kernel_ID)).index_select(2, torch.tensor(kernel_height))
            basekernel_weights = np.squeeze(basekernel_weights)
            h, angles, f_hz = BP_coefficients(basekernel_weights, spec_dict['srate'], spec_dict['flim'])
            
            # Plot baseline comparison
            ax = BP_plot(ax, i, h, angles, f_hz, spec_dict, str(f_id), [min_h, max_h, min_a, max_a], 1)
            ax[i].legend(handles=legend1_elements, loc=spec_dict['loc'], fontsize=spec_dict['font'] - 6)
    
    # Remove unused axes in grid layout
    for i in range(filters, int(nfig**2 - 1)):
        fig.delaxes(ax[i])
    
    return fig, ax

#### STEM PLOT ####
def weights_plot(gs, i, kernel_weights, fID_string, kernel_ID, spec_dict):
    """
    Creates a pair of plots for visualizing kernel weights: a stem plot and an image plot.
    
    Parameters:
    -----------
    - gs (gridspec.GridSpec): The grid specification for the figure layout.
    - i (int): The index of the current filter being plotted.
    - kernel_weights (torch.Tensor): The weights of the kernel to be plotted.
    - fID_string (str): A string identifier for the filter.
    - kernel_ID (int): ID of the kernel within the filter selected.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.

    Returns:
    --------
    - None: The function modifies the provided axis `ax`.
    """

    # Create the stem plot for kernel weights
    ax0 = plt.subplot(gs[i * 2])
    stem = ax0.stem(torch.squeeze(kernel_weights))  # Plot the kernel weights
    stem.markerline.set_markersize(spec_dict['s'])  # Set marker size
    ax0.set_title(f'Filter$_{{{fID_string}}}$', fontsize=spec_dict['font'], pad=2)  # Title with filter ID
    
    # Adjust x-axis ticks (empty by default)
    ax0.set_xticks([]) if spec_dict['channels'] is None else ax0.set_xticks([])
    ax0.set_ylabel('Kernel Weights')
    ax0.set_xlabel('Kernel Length')

    extended_lims = get_extended_lims(spec_dict['vlim']) 
    
    ax0.set_ylim(extended_lims[0],extended_lims[1])
    ax0.set_yticks(np.linspace(spec_dict['vlim'][0], spec_dict['vlim'][1], spec_dict['ytick']))
    ax0.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))

    # Create the imshow plot for kernel weights (2D representation)
    ax1 = plt.subplot(gs[i * 2 + 1], sharex=ax0)
    ax1.imshow(torch.squeeze(kernel_weights, dim=(0, 1)), cmap=spec_dict['cmap'], aspect='auto',
               interpolation='nearest', vmin=spec_dict['vlim'][0], vmax=spec_dict['vlim'][1])  # 2D heatmap
    ax1.set_xticks([])  # Hide x-ticks
    ax1.set_yticks([])  # Hide y-ticks


def weights_temporalkernels(Mdl_weights, layer,
                            spec_dict, 
                            filter_ID=None, kernel_ID=0, kernel_height=0):
    """
    Visualizes the kernel weights of a specific layer in the model, with both stem plots 
    and heatmaps.
    Optionally, a specific filter or filters can be plotted.
    
    Parameters:
    -----------
    - Mdl_weights (dict): A dictionary containing the model's weights, 
    with keys corresponding to layer names.
    - layer (str): The name of the layer for which to plot the weights.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - filter_ID (int or list, optional): ID(s) of the filters to plot. Defaults to None (all filters).
    - kernel_ID (int, optional): ID of the kernel within the filter selected (default is 0).
    - kernel_height (int, optional): Height index within the kernel to select the relevant slice (default is 0).
    
    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    """
    
    # Retrieve the weights for the specified layer
    weights = Mdl_weights[layer + '.weight']
    
    # Determine which filters to plot
    if filter_ID is None:
        filters, fm, height, width = weights.shape
        filter_IDs = range(filters)
    else:
        filter_IDs = filter_ID if isinstance(filter_ID, list) else [filter_ID]
        filters = len(filter_IDs)
    
    # Set up the figure and grid layout
    fig = plt.figure(figsize=(spec_dict['figdim'][0]*2, spec_dict['figdim'][1] * filters))
    gs  = gridspec.GridSpec(2*filters, 1, height_ratios=[2, 0.25]*filters, hspace=0.35)

    # Set the vertical limits for the plot based on the kernel weights
    symmetric_lim     = np.max([np.abs(weights.min().numpy()), np.abs(weights.max().numpy())])
    spec_dict['vlim'] = [-symmetric_lim, symmetric_lim]
    
    # Plot each filter's kernel weights
    for i, f_id in enumerate(filter_IDs):
        kernel_weights = weights.index_select(0, torch.tensor(f_id)).index_select(1, torch.tensor(kernel_ID))
        fID_string = str(f_id)  # Convert filter ID to string for plotting
        weights_plot(gs, i, kernel_weights, fID_string, kernel_ID, spec_dict)
 
    return fig

###################################
####   ACTIVATION FOR 2 LAYER   ###
###################################
def temporal_FM_PSD_layer2(input_window, 
                           Mdl, layer,
                           spec_dict, inout_labels,
                           batch_ID, FM_ID=None):
    """
    Computes and visualizes the Power Spectral Density (PSD) of the feature maps of a specified 
    layer in the model for a given input window and batch index. 
    This function computes the PSD for one or more feature maps and displays the results using
    a heatmap.
    
    Parameters:
    -----------
    - input_window (torch.Tensor): The input tensor containing the data,
    with dimensions [batch, 1, channels, timesteps].
    - Mdl (torch.nn.Module): The model to apply to the input window for generating convoluted output.
    - layer (str): The name of the layer for which to plot the weights.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - inout_labels (list of lists): A list containing true and predicted labels for all the samples in input_window.
    - batch_ID (int): Index of the batch to analyze.
    - FM_ID (int or list, optional): IDs of feature maps to plot. Defaults to None (all FMs).
    
    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    """

    # Ensure that batch_ID is a scalar if it's provided as a list
    batch_ID = batch_ID[0] if isinstance(batch_ID, list) else batch_ID
    
    # Get the convoluted output from the model for the specified layer and input window
    convoluted_output = out_activation(Mdl, layer, input_window)
    
    # Determine which feature maps to visualize
    if FM_ID is None:
        batch, fm, chan, timestep = convoluted_output.shape
        fm_IDs = range(fm)
    else:
        fm_IDs = FM_ID if isinstance(FM_ID, list) else [FM_ID]
        fm = len(fm_IDs)
    
    # Get the PSD for the first feature map
    time_w = torch.squeeze(convoluted_output.index_select(0, torch.tensor(batch_ID)).index_select(1, torch.tensor(0)))
    f, Pxx = get_Pxx(time_w, spec_dict['srate'])
    
    # Create a tensor to store PSD values for all selected feature maps
    Pxx_tensor = np.zeros((fm, Pxx.shape[0]))
    
    # Calculate and store the PSD for each feature map
    for i, fm_id in enumerate(fm_IDs):
        time_w = torch.squeeze(convoluted_output.index_select(0, torch.tensor(batch_ID)).index_select(1, torch.tensor(fm_id)))
        f, Pxx = get_Pxx(time_w, spec_dict['srate'])
        Pxx_tensor[i, :] = Pxx

    # Get the frequency index based on the frequency limits
    f_index = get_freq_index(f, spec_dict['flim'])

    spec_dict['vlim'] = get_vlim(convoluted_output,spec_dict['srate'],spec_dict['flim'], spec_dict['lognorm'])
    extended_lims = get_extended_lims(spec_dict['vlim'])
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(spec_dict['figdim'][0], spec_dict['figdim'][1]))  # Set figure size    
    # Plot the PSD using imshow (heatmap)
    im = ax.imshow(data_transform(Pxx_tensor[:, f_index[0]:f_index[1]],spec_dict['lognorm']), aspect='auto',
                   cmap=spec_dict['cmap'], vmin=extended_lims[0], vmax=extended_lims[1])

    # Add colorbar to the plot
    unit = get_PSDunit(spec_dict['lognorm'], unit=None, power=False)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(f'PSD$_{{{unit}}}$', fontsize=spec_dict['font'] - 2)  # Set colorbar label [TO DO UNIT]
    cbar.ax.tick_params(labelsize=spec_dict['font'] - 4)  # Adjust colorbar tick size
    cbar.set_ticks(np.linspace(spec_dict['vlim'][0], spec_dict['vlim'][1], spec_dict['cbartick']))  # Set colorbar ticks

    # Set title and axis labels
    ax.set_title(f'Batch Index: {batch_ID}| {inout_labels[0][batch_ID]} -> {inout_labels[1][batch_ID]}', 
                 fontsize=spec_dict['font'])
    
    # Set x-axis ticks and labels based on frequency range
    ax.set_xticks(np.arange(f_index[0], f_index[1], spec_dict['groupby'][0]) - f_index[0])
    ax.set_xticklabels([f"{value:.2f}" for value in f[f_index[0]:f_index[1]:spec_dict['groupby'][0]]], 
                       rotation=spec_dict['rotation'], fontsize=spec_dict['font'] - 4)
    ax.set_xlabel('$f_{[Hz]}$', fontsize=spec_dict['font'] - 2)

    # Set y-axis ticks and labels for feature map IDs
    ax.set_yticks(np.arange(0, fm, spec_dict['groupby'][1]))
    ax.set_yticklabels(fm_IDs[::spec_dict['groupby'][1]], fontsize=spec_dict['font'] - 4)
    ax.set_ylabel('Feature Map ID', fontsize=spec_dict['font'] - 2)
    
    # Turn off the grid for the plot
    ax.grid(False)
    
    return fig

def temporal_FM_layer2(input_window,
                       Mdl, layer,
                       spec_dict,
                       batch_ID, FM_ID=None):
    """
    This function processes a given input window through a model and specified layer, 
    generates the convoluted output, and plots the result using MNE's Raw data plot functionality.

    Parameters:
    -----------
    - input_window (torch.Tensor): The input tensor containing the data,
    with dimensions [batch, 1, channels, timesteps].
    - Mdl (torch.nn.Module): The model to apply to the input window for generating convoluted output.
    - layer (str): The name of the layer for which to plot the weights.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - batch_ID (int): Index of the batch to analyze.
    
    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    """
    
    # Set the MNE backend for plotting to 'matplotlib' (avoids interactive browser-based plots)
    mne.viz.set_browser_backend(spec_dict['backend'])
    logging.getLogger('mne').setLevel(logging.WARNING)
    
    # Ensure that batch_ID is a scalar if it's provided as a list (otherwise, use directly)
    batch_ID = batch_ID[0] if isinstance(batch_ID, list) else batch_ID
    
    # Get the convoluted output from the model for the specified layer and input window
    convoluted_output = out_activation(Mdl, layer, input_window)

    # Determine which feature maps to visualize
    if FM_ID is None:
        batch, fm, chan, timestep = convoluted_output.shape
        fm_IDs = range(fm)
    else:
        fm_IDs = FM_ID if isinstance(FM_ID, list) else [FM_ID]
        fm = len(fm_IDs)
    
    # Select the batch index and remove any extra dimensions (squeeze the tensor)
    data = torch.squeeze(convoluted_output.index_select(0, torch.tensor(batch_ID)).index_select(1,torch.tensor(fm_IDs)))

    # Generate FM names if not provided in spec_dict, otherwise use the ones from spec_dict
    if spec_dict['FMnames'] is None:
        FM_names = ['FM {}'.format(i) for i in fm_IDs]
    else:
        if len(spec_dict['FMnames'])==fm:
            FM_names = spec_dict['FMnames']
        else:
            raise ValueError("Length of spec_dict['FMnames'] is not equal to the FM_ID selected.")
    
    # Create MNE info structure for the raw data
    info = mne.create_info(ch_names=FM_names, sfreq=spec_dict['srate'])
    
    # Create a RawArray object in MNE from the processed data
    raw = mne.io.RawArray(data.detach().numpy(), info)
    
    # Generate the figure by plotting the raw data with automatic scaling
    fig = raw.plot(scalings='auto', verbose=False)
    
    return fig
    
###################################
####    WEIGHTS FOR 2 LAYER     ###
###################################
def get_weights(weights, filter_ID, kernel_ID):
    """
    Extracts a subset of weights based on specified filter and feature map (FM) IDs.

    This function selects and returns a subset of the `weights` tensor by indexing the weights 
    along the first and second dimensions based on the provided `FM_ID` and `filter_ID`. 
    If `FM_ID` or `filter_ID` is `None`, it skips the respective indexing for that dimension.

    Parameters:
    -----------
    - weights (torch.Tensor): The tensor containing the model weights. 
    It is expected to have at least two dimensions, where the first dimension corresponds
    to feature maps and the second corresponds to filters.
    - filter_ID (None, int, list, or tuple): 
        - If `None`, no filtering is applied along the second dimension (filters).
        - If an integer, it selects the specified filter index.
        - If a list or tuple, it selects the specified filters indexed by `filter_ID`.
    - FM_ID (None, int, list, or tuple):
        - If `None`, no filtering is applied along the first dimension (feature maps).
        - If an integer, it selects the specified feature map index.
        - If a list or tuple, it selects the specified feature maps indexed by `FM_ID`.

    Returns:
    --------
    - weights_sel (torch.Tensor): A subset of the `weights` tensor, 
    indexed by the `FM_ID` and `filter_ID`. 
    If both are `None`, the original `weights` tensor is returned without any modifications.
    """
    
    if filter_ID is not None:
        weights_sel = weights.index_select(0, torch.tensor(filter_ID))
        
    if kernel_ID is not None:
        weights_sel = weights.index_select(1, torch.tensor(kernel_ID))

    if (kernel_ID is None) and (filter_ID is None):
        weights_sel = weights.copy()
        
    return weights_sel
    
def get_strings(filter_ID, kernel_ID, i, j):
    """
    Generates a list of string representations for filter and kernels IDs based
    on provided inputs.

    This function returns a list of two strings representing the filter ID and feature map ID.
    The values are determined by the `filter_ID` and `kernel_ID` inputs, 
    or by the provided indices `i` and `j` if the IDs are not provided.

    Parameters:
    -----------
    - filter_ID (None, int, list, or tuple): 
        - If `None`, the function uses the index `i` as the filter ID.
        - If an integer, it uses the value as the filter ID.
        - If a list or tuple, it uses the `i`-th element as the filter ID (if available).
    - kernel_ID (None, int, list, or tuple): 
        - If `None`, the function uses the index `j` as the kernel ID.
        - If an integer, it uses the value as the kernel ID.
        - If a list or tuple, it uses the `j`-th element as the kernel ID (if available).
    - i (int): Index used for the filter ID if `filter_ID` is `None` or a list/tuple.
    - j (int): Index used for the kernel ID if `kernel_ID` is `None` or a list/tuple.

    Returns:
    -------
    - list[str]: A list containing two strings: `[filter_ID_string, kernel_ID_string]`.
    """

    fID  = str(filter_ID[i] if isinstance(filter_ID, (list, tuple)) else filter_ID) if filter_ID is not None else str(i)
    kID = str(kernel_ID[j]  if isinstance(kernel_ID, (list, tuple)) else kernel_ID) if kernel_ID is not None else str(j)
    return [fID, kID]

def spatialkernels_plot(ax, ax_idx1, ax_idx2, evoked_data, spec_dict, string_ID):
    """
    Plots a spatial topographic map of the evoked data on the specified axes.

    This function plots a topographic map (using `evoked_data`) on two axes of a given figure. 
    It handles axes selection, adjusts tick and label properties, and sets appropriate titles 
    based on the provided `spec_dict` and `string_ID`. The topomap is rendered with a colorbar 
    and specific limits based on the `vlim` parameter from `spec_dict`.

    Parameters:
    ----------
    - ax (matplotlib.Axes): The axes on which the topographic map will be plotted.
    - ax_idx1 (list or tuple): The index of the first axis. 
    The first element can be `None` to select a single axis.
    - ax_idx2 (list or tuple): The index of the second axis. Similar to `ax_idx1`, 
    the first element can be `None`.
    - evoked_data (mne.EvokedArray): The evoked data object used to plot the topomap. 
    It contains the data to be visualized.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - string_ID (list): A list containing the string representations of the filter and kernel IDs 
    (e.g., `['filter_ID', 'kernel_ID']`).

    Returns:
    --------
    - None: This function modifies the axes and does not return anything.
    """
    
    axes1 = ax[ax_idx1[1]] if ax_idx1[0] is None else ax[ax_idx1[0], ax_idx1[1]]
    axes2 = ax[ax_idx2[1]] if ax_idx2[0] is None else ax[ax_idx2[0], ax_idx2[1]]
    
    # Plot the topomap
    evoked_data.plot_topomap(show=False, axes=(axes1, axes2), show_names=True, 
                             cmap=spec_dict['cmap'], cbar_fmt='%.1f', scalings=1,
                             vlim=get_extended_lims(spec_dict['vlim']), contours=spec_dict['contours'])

    # Adjust axes and labels
    axes2.set_yticks(np.linspace(spec_dict['vlim'][0], spec_dict['vlim'][1], spec_dict['cbartick']))
    axes2.tick_params(labelsize=spec_dict['font']-6)
    axes2.set_title('$w_{ij}$', fontsize=spec_dict['font']-4)
    for text in axes1.texts:
        text.set_fontsize(spec_dict['font']-8)
        text.set_fontweight('bold')

    # Set title for axes1
    title = f"Filter$_{{{string_ID[0]}}}$, Input FM$_{{{string_ID[1]}}}$" if spec_dict['filternames'] is None else f"Scalp: {spec_dict['filternames'][int(ax_idx1[1] / 2)]}"
    axes1.set_title(title, fontsize=spec_dict['font'])

def spatialkernels_scalp(Mdl_weights, layer, 
                         spec_dict, 
                         filter_ID=0, kernel_ID=0, kernel_width=0, 
                         MdlBase_weights=None):
    """
    Visualizes the spatial kernels of a specific layer in a model, plotting the kernel weights 
    for each filter and feature map. 
    The function supports different normalization methods (e.g., 'softmax', 'nothing', 'abs')
    and optionally compares to baseline weights.
    
    Parameters:
    -----------
    - Mdl_weights (dict): A dictionary containing the model's weights, 
    with keys corresponding to layer names.
    - layer (str): The name of the layer for which to plot the weights.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - filter_ID (int or list, optional): ID(s) of the filters to plot. Defaults to None (all filters).
    - kernel_ID (int, optional): ID of the kernel within the filter selected (default is 0).
    - kernel_width (int, optional): Width index within the kernel to select the relevant slice (default is 0).
    - MdlBase_weights (dict, optional): A dictionary containing the baseline (pre-training)
    model weights for comparison. Default is None.
    
    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    """
    
    info    = mne.create_info(ch_names=spec_dict['channels'], sfreq=spec_dict['srate'], ch_types='eeg')
    montage = mne.channels.make_standard_montage(spec_dict['montage'])

    weights = Mdl_weights[layer + '.weight']

    # Apply normalization based on the specified method
    if spec_dict['norm'] == 'softmax':
        weights = torch.softmax(weights, -2)
        if MdlBase_weights:
            baseweights = torch.softmax(MdlBase_weights[layer + '.weight'], -2)
            spec_dict['vlim'] = (spec_dict['min_softmax'], 
                                 np.max([baseweights.max().item(), weights.max().item()]))
        else:
            spec_dict['vlim'] = (spec_dict['min_softmax'], weights.max().item())
        spec_dict['cmap'] = spec_dict['cmap'][1]
        
    elif spec_dict['norm'] == 'nothing':
        if MdlBase_weights:
            baseweights = MdlBase_weights[layer + '.weight']
            spec_dict['vlim'] = (np.min([baseweights.min().item(), weights.min().item()]),
                                 np.max([baseweights.max().item(), weights.max().item()]))
        else:
            spec_dict['vlim'] = (weights.min().item(), weights.max().item())
        spec_dict['cmap'] = spec_dict['cmap'][0]
        
    elif spec_dict['norm'] == 'abs':
        weights = torch.abs(weights)
        if MdlBase_weights:
            baseweights = torch.abs(MdlBase_weights[layer + '.weight'])
            spec_dict['vlim'] = (0, np.max([baseweights.max().item(), weights.max().item()]))
        else:
            spec_dict['vlim'] = (0, weights.max().item())
        spec_dict['cmap'] = spec_dict['cmap'][1]

    # Select specific FM and filter if provided
    weights = get_weights(weights, filter_ID, kernel_ID)
    
    filters, kernels, height, width = weights.shape
    fig, ax = plt.subplots(kernels, filters * 2, 
                           figsize=(spec_dict['figdim'][0] * filters, spec_dict['figdim'][1] * kernels),
                           gridspec_kw={'width_ratios': [0.9, 0.1] * filters}, constrained_layout=True)
    fig.suptitle(spec_dict['title'], fontsize=spec_dict['font'] + 2)

    # Plot the spatial kernels
    for j in range(kernels):
        for i in range(filters):
            data_vector = weights[i, j, :, kernel_width]
            evoked_data = mne.EvokedArray(data_vector[:, np.newaxis], info)
            evoked_data.set_montage(montage)
            string_ID = get_strings(filter_ID, kernel_ID, i, j)

            if kernels == 1 and filters > 1:
                spatialkernels_plot(ax, [None, 2 * i], [None, 2 * i + 1], evoked_data, spec_dict, string_ID)
            elif kernels > 1 and filters == 1:
                spatialkernels_plot(ax, [j, 0], [j, 1], evoked_data, spec_dict, string_ID)
            else:
                spatialkernels_plot(ax, [j, 2 * i], [j, 2 * i + 1], evoked_data, spec_dict, string_ID)

    return fig

###################################
####   ACTIVATION FOR FLATTEN   ###
###################################
def flatten_fm(input_window, 
               Mdl, layer, 
               spec_dict, inout_labels, 
               batch_ID):
    """
    Visualizes the activations of feature maps from a specific layer of the model for one or
    more batch indices. 
    The activations are displayed as images for each batch, with each feature map in the layer
    visualized as a row in the image.
    
    Parameters:
    -----------
    - input_window (torch.Tensor): The input tensor containing the data,
    with dimensions [batch, 1, channels, timesteps].
    - Mdl_weights (dict): A dictionary containing the model's weights, 
    with keys corresponding to layer names.
    - layer (str): The name of the layer for which to plot the weights.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - inout_labels (list of lists): A list containing two lists: one for the input labels and 
    one for the output labels.
    - batch_ID (int): Index of the batch to analyze.
    
    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    """

    # Get the convoluted output (activations) from the specified layer of the model
    convoluted_output = out_activation(Mdl, layer, input_window)

    spec_dict['vlim'] = (convoluted_output.min().numpy(),convoluted_output.max().numpy())

    # Number of feature maps (neurons) in the layer
    neurons_feat = convoluted_output.shape[1]
    
    # Ensure that batch_ID is a scalar if it's provided as a list
    batch_ID = batch_ID[0] if isinstance(batch_ID, list) else batch_ID
    
    # Set up the figure and axes for visualization
    fig, ax = plt.subplots(1, 1, figsize=(spec_dict['figdim'][0], spec_dict['figdim'][1]),
                                   constrained_layout=True)
    
    batch_activ = convoluted_output.index_select(0, torch.tensor(batch_ID)).numpy()

    # Plot the activations for the current batch
    ax.imshow(batch_activ, aspect='auto', cmap=spec_dict['cmap'], interpolation='nearest',
              vmin=spec_dict['vlim'][0], vmax=spec_dict['vlim'][1])
    
    # Set x-axis ticks and labels (showing feature map indices)
    ax.set_xticks(np.arange(0, neurons_feat, spec_dict['groupby']))
    ax.set_xticklabels(labels=['FM-' + str(j) for j in range(0, neurons_feat, spec_dict['groupby'])],
                             rotation=spec_dict['rotation'], fontsize=spec_dict['font'] - 4)
    
    # Remove y-axis ticks (not needed for feature map visualization)
    ax.set_yticks([])
    
    # Set the title of the plot with batch index and labels
    ax.set_title(f'Batch Index: {batch_ID}| {inout_labels[0][batch_ID]} -> {inout_labels[1][batch_ID]}',fontsize=spec_dict['font'])
    
    # Draw vertical lines between feature maps for visual clarity
    for j in range(0, neurons_feat, 1):
        ax.axvline(j - 0.5, color='k', linewidth=spec_dict['linew'], alpha=0.7)

    # Disable grid lines
    ax.grid(False)
            
    return fig

def flattenCorrelationPSD(input_window, 
                          Mdl, layer,
                          spec_dict):
    """
    This function computes the correlation between the frequency band power 
    from a given EEG signal and the output of a model's layer, then visualizes 
    the relationship with scatter plots and linear regression.

    It computes the power spectral density (PSD) of the input EEG signal for 
    specified frequency bands, then compares the band powers to the model's 
    layer output. Linear regression is applied, and plots of these relationships 
    are displayed, showing the adjusted R-squared and p-values.

    Parameters:
    -----------
    - input_window (torch.Tensor): The input tensor containing the data,
    with dimensions [batch, 1, channels, timesteps].
    - Mdl_weights (dict): A dictionary containing the model's weights, 
    with keys corresponding to layer names.
    - layer (str): The name of the layer for which to plot the weights.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.

    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    """
    
    # Extract the frequency band keys from the specification dictionary
    band_keys = list(spec_dict['bands'].keys())
    
    # Compute the Power Spectral Density (Pxx) for the first sample
    f, Pxx = get_Pxx(input_window[0, :, :], srate=spec_dict['srate'])
    
    # Initialize an array to store the power content for each frequency band
    band_content = np.zeros((input_window.shape[0], len(band_keys)))
    
    # Loop over each sample to calculate the band power content
    for i in range(input_window.shape[0]):
        f, Pxx = get_Pxx(input_window[i, :, :], srate=spec_dict['srate'])
    
        for j in range(len(band_keys)):
            # Get the frequency index for the current frequency band
            findex = get_freq_index(f, spec_dict['bands'][band_keys[j]])
            # Calculate the frequency range for normalization
            frange = (spec_dict['bands'][band_keys[j]][1] - spec_dict['bands'][band_keys[j]][0])
            # Compute the log10 of the average power in the current band using trapezoidal integration
            #bandc = spec_dict['norm']*np.log10(np.mean(np.trapz(Pxx[findex[0]:findex[1], :].T, dx=(f[1] - f[0])) / frange))
            bandc = data_transform(np.mean(np.trapz(Pxx[findex[0]:findex[1], :].T, dx=(f[1] - f[0]))/1),spec_dict['lognorm'])
            band_content[i, j] = bandc

    unit = get_PSDunit(spec_dict['lognorm'], unit=None, power=True)
    # Get the model output from the specified layer
    layer = 'encoder'
    convoluted_output = out_activation(Mdl, layer, input_window)
    
    # Normalize the model output
    convoluted_output = convoluted_output
    
    # Set up the figure layout for the subplots (one for each frequency band)
    nfig = int(np.ceil(np.sqrt(len(band_keys))))  # Calculate the number of rows and columns for subplots
    fig, axs = plt.subplots(nfig - 1, nfig + 1, 
                            figsize=(spec_dict['figdim'][0] * (nfig + 1), spec_dict['figdim'][1] * (nfig - 1)),
                            constrained_layout=True)
    axs = axs.flatten()  # Flatten the 2D array of axes to easily index
    
    # Lists to store p-values and adjusted R-squared values
    p_values = []
    r_values = []
    adj_rvalues = []
    
    # Loop through each band and plot the scatter plot with regression line
    for b in range(len(band_keys)):
        # Data for the current band
        x = band_content[:, b]
        y = convoluted_output[:, b]
    
        # Perform linear regression and get the p-value and R-squared value
        slope, intercept, r_value, p_value, slope_err = linregress(x, y)
        p_values.append(p_value)
        r_values.append(r_value)
        
        # Compute adjusted R-squared
        adj_rvalues.append(adjusted_R2(r_value, len(x), 1))
    
        # Create scatter plot and regression line
        axs[b].tick_params(axis='x', labelsize=spec_dict['font'] - 4)
        axs[b].tick_params(axis='y', labelsize=spec_dict['font'] - 4)
        axs[b].scatter(x, y, s=spec_dict['s'], edgecolors='none')
        equation = f"y = {slope:.1f}x {'+' if intercept >= 0 else '-'} {abs(intercept):.1f}"
        axs[b].plot(x, slope * x + intercept, color='red', label=equation)

        # Adjust the axis limits to provide some padding around the data
        range_x = [np.min(band_content), np.max(band_content)]
        range_y = [convoluted_output.min().numpy(), convoluted_output.max().numpy()]
        
        range_x = get_extended_lims(range_x)
        range_y = get_extended_lims(range_y)
        
        axs[b].set_xlim(range_x[0], range_x[1])
        axs[b].set_ylim(range_y[0], range_y[1])

    # Perform multiple test correction using p-values
    corrected_res = multitest.multipletests(p_values, alpha=spec_dict['pval'], method=spec_dict['method'])
    corr_pvalues = corrected_res[1]
    
    # Annotate the plots with the corrected p-values, R-squared, and labels
    for b in range(len(band_keys)):
        if corr_pvalues[b] < 0.001:  # Bonferroni corrected
            pstring = '<.001'
        else:
            pstring = f'={corr_pvalues[b]:.2f}'.split(".")[1]
        
        # Set the axis labels and title with the p-value and adjusted R-squared
        axs[b].set_xlabel(f'Data: $\\mathcal{{F}}_X$$(${band_keys[b]}$)$'+f'$~{unit}$', 
                          fontsize=spec_dict['font'] - 2)
        axs[b].set_ylabel(f'Flatten Activation: $\\mathcal{{F}}$$(${band_keys[b]}$)$'+f'$~{unit}$', 
                          fontsize=spec_dict['font'] - 2)
        title = f'$\\rho({len(x)-2})=.{f"{r_values[b]:.2f}".split(".")[1]},~p{pstring},~Adj.R^2=.{f"{adj_rvalues[b]:.2f}".split(".")[1]}$'
        axs[b].set_title(title, fontsize=spec_dict['font']-4)
        axs[b].legend(loc=spec_dict['loc'], fontsize=spec_dict['font'] - 6)
    
    # Remove any extra subplots that are not needed
    for i in range(len(band_keys), int(nfig**2 - 1)):
        fig.delaxes(axs[i])

    return fig

###################################
####       DENSE WEIGHTS        ###
###################################
def denseweights_plot(Mdl_weights, layer, 
                      spec_dict):
    """
    Plots the weights and the class with the greatest weights for a dense layer in a model.
    
    Parameters:
    -----------
    - Mdl_weights (dict): A dictionary containing the model's weights, 
    with keys corresponding to layer names.
    - layer (str): The name of the layer for which to plot the weights.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    
    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    - mask (np.ndarray): A mask indicating the class with the greatest weight for each neuron.
    """
    
    # Number of classes and neurons in the dense layer
    nb_classes = len(spec_dict['classlabels'])
    out_neurons, dense_neurons = Mdl_weights[layer + '.weight'].shape
    
    # Determine which plots to create based on plot_type
    plot_type = spec_dict.get('plot_type', 'both')
    
    if plot_type == 'first':  # Only plot the weights
        fig, ax = plt.subplots(1, 1, figsize=(spec_dict['figdim'][0], spec_dict['figdim'][1]))
        ax = [ax]  # Wrap single axis in list for consistency
        
    elif plot_type == 'second':  # Only plot the greatest weights class
        fig, ax = plt.subplots(1, 1, figsize=(spec_dict['figdim'][0], spec_dict['figdim'][1]))
        ax = [ax]  # Wrap single axis in list for consistency
        
    else:  # Plot both the weights and the greatest weights class
        fig, ax = plt.subplots(2, 1, figsize=(spec_dict['figdim'][0], spec_dict['figdim'][1]*2),
                               constrained_layout=True)
    
    # Plot the weights if 'first' or 'both' is selected
    if plot_type in ['first', 'both']:
        mask = []
        weights = Mdl_weights[layer + '.weight']

        symmetric_lim     = np.max([np.abs(weights.min().numpy()), np.abs(weights.max().numpy())])
        spec_dict['vlim'] = [-symmetric_lim, symmetric_lim]
        extended_lims = get_extended_lims(spec_dict['vlim'])
        
        # Plot the weights heatmap
        im = ax[0].imshow(weights, aspect='auto', cmap=spec_dict['cmap'], interpolation='nearest',
                           vmin=extended_lims[0], vmax=extended_lims[1])
        
        # Set y-ticks (class labels)
        if nb_classes <= 2:
            ax[0].set_yticks([0], 
                             labels=[spec_dict['classlabels'][0]], fontsize=spec_dict['font'] - 4)
        else:
            ax[0].set_yticks(np.arange(0, nb_classes), 
                             labels=spec_dict['classlabels'], fontsize=spec_dict['font'] - 4)

        ax[0].set_title(spec_dict['title'], fontsize=spec_dict['font'])
        ax[0].set_xlabel('Neuron ID', fontsize=spec_dict['font'] - 2)
        
        # Set x-ticks for neuron IDs
        if spec_dict['xticks'] is None:
            ax[0].set_xticks(np.arange(0, dense_neurons, spec_dict['groupby']))
            ax[0].set_xticklabels(np.arange(0, dense_neurons, spec_dict['groupby']), fontsize=spec_dict['font'] - 4)
        else:
            ax[0].set_xticks(np.arange(0, len(spec_dict['xticks']), 1))
            ax[0].set_xticklabels(spec_dict['xticks'], fontsize=spec_dict['font'] - 4)

        # Add colorbar to the plot
        cbar = fig.colorbar(im, ax=ax[0])
        cbar.ax.set_ylabel('$w_{ij}$', fontsize=spec_dict['font'] - 4)
        cbar.ax.tick_params(labelsize=spec_dict['font'] - 2)
        cbar.set_ticks(np.linspace(spec_dict['vlim'][0], spec_dict['vlim'][1], spec_dict['cbartick']))
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Plot the greatest weights class if 'second' or 'both' is selected
    if plot_type in ['second', 'both']:
        unique_colori = ['white', 'red', 'blue', 'green']
        cmap = ListedColormap(unique_colori)  # Custom colormap for the mask
        
        weights = Mdl_weights[layer + '.weight']
        mask = -np.ones((nb_classes, dense_neurons), dtype=int)
        
        # Generate the mask based on the greatest weights for each neuron
        if nb_classes <= 2:
            for i in range(dense_neurons):
                mask[0, torch.squeeze(weights) <= 0] = 0
                mask[1, torch.squeeze(weights) > 0] = 1
        else:
            for i in range(dense_neurons):
                mask[np.argmax(weights, axis=0)[i], i] = np.argmax(weights, axis=0)[i]
        
        # Plot the greatest weight class mask
        if plot_type == 'second':  # Use the first axis for the second plot
            ax[0].imshow(mask, aspect='auto', cmap=cmap, interpolation='nearest')
            ax[0].set_title('Greatest Weight', fontsize=spec_dict['font'])
            ax[0].set_yticks(np.arange(0, nb_classes))
            ax[0].set_yticklabels(spec_dict['classlabels'], fontsize=spec_dict['font'] - 4)
            ax[0].set_xlabel('Neuron ID', fontsize=spec_dict['font'] - 2)
            
            # Set x-ticks for the second plot
            if spec_dict['xticks'] is None:
                ax[0].set_xticks(np.arange(0, dense_neurons, spec_dict['groupby']))
                ax[0].set_xticklabels(np.arange(0, dense_neurons, spec_dict['groupby']), fontsize=spec_dict['font'] - 4)
            else:
                ax[0].set_xticks(np.arange(0, len(spec_dict['xticks']), 1))
                ax[0].set_xticklabels(spec_dict['xticks'], fontsize=spec_dict['font'] - 4)
                
        else:  # Plot on the second subplot when both plots are selected
            ax[1].imshow(mask, aspect='auto', cmap=cmap, interpolation='nearest')
            ax[1].set_title('Greatest Weight', fontsize=spec_dict['font'])
            ax[1].set_yticks(np.arange(0, nb_classes))
            ax[1].set_yticklabels(spec_dict['classlabels'], fontsize=spec_dict['font'] - 4)
            ax[1].set_xlabel('Neuron ID', fontsize=spec_dict['font'] - 2)
            
            # Set x-ticks for the second plot
            if spec_dict['xticks'] is None:
                ax[1].set_xticks(np.arange(0, dense_neurons, spec_dict['groupby']))
                ax[1].set_xticklabels(np.arange(0, dense_neurons, spec_dict['groupby']), fontsize=spec_dict['font'] - 4)
            else:
                ax[1].set_xticks(np.arange(0, len(spec_dict['xticks']), 1))
                ax[1].set_xticklabels(spec_dict['xticks'], fontsize=spec_dict['font'] - 4)

    return fig, mask

###################################
####   EMBEDDING VISUALIZATION  ###
###################################

#### EMBEDDING X SPLIT: TRAIN-VALIDATION-TEST ####
def embedding_split2D(embedding, labels, 
                      spec_dict, 
                      ax=None):
    """
    Visualizes a 2D embedding with scatter plots for the TEST, VAL, and TRAIN sets, 
    and optionally adds KDE contours for each set.

    Parameters:
    -----------
    - embedding (np.ndarray): 2D array of shape (n_samples, 2) containing the embedding points to plot.
    - labels (list of np.ndarray): A list containing three arrays 
    for TEST, VAL, and TRAIN labels, respectively.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - ax (matplotlib.axes.Axes, optional): The axes on which to plot. 
    If None, a new figure and axes are created.

    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    - ax (matplotlib.axes.Axes): The axes containing the plot.
    """

    # Create a new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(spec_dict['figdim'][0], spec_dict['figdim'][1]))
    else:
        fig = None  # No need to create a new figure
    
    # Calculate sample sizes for TEST, VAL, and TRAIN sets
    sample_sizes = [len(labels[0]), len(labels[1]), len(labels[2])]

    idxs = {'TEST':  [0, len(labels[0])],
            'VAL':   [len(labels[0]), len(labels[0]) + len(labels[1])],
            'TRAIN': [len(labels[0]) + len(labels[1]), -1]}
    
    # Cumulative indices for slicing the embedding data
    cumulative_sizes = np.cumsum([0] + sample_sizes)
    
    # Plot the scatter points for each sample set (TEST, VAL, TRAIN)
    legend_elements = []
    for i in range(len(spec_dict['sets'])):
        alpha = spec_dict['alpha_trainval'] if spec_dict['sets'][i] != 'TEST' else spec_dict['alpha_test']
        size  = spec_dict['s_trainval']     if spec_dict['sets'][i] != 'TEST' else spec_dict['s_test']
        mask  = [k == spec_dict['sets'][i] for k in list(idxs.keys())]
        color = [spec_dict['colors'][j] for j, m in enumerate(mask) if m][0]

        ax.scatter(embedding[idxs[spec_dict['sets'][i]][0]:idxs[spec_dict['sets'][i]][1], 0],
                   embedding[idxs[spec_dict['sets'][i]][0]:idxs[spec_dict['sets'][i]][1], 1],
                   color=color,
                   alpha=alpha,
                   marker=spec_dict['marker'],
                   s=size)
        legend_elements.append(Line2D([0], [0], marker=spec_dict['marker'], color=color, alpha=alpha,
                                      markersize=size, label=spec_dict['sets'][i], linestyle='None'))
    
    # Add 2D KDE contours if specified in the configuration
    if 'dim' in spec_dict:
        for i in range(len(spec_dict['sets'])):
            mask = [k == spec_dict['sets'][i] for k in list(idxs.keys())]
            color = [spec_dict['colors_lines'][j] for j, m in enumerate(mask) if m][0]
            
            X, Y, Z, threshold_value = compute_2D_KDE(embedding, 
                                                      [idxs[spec_dict['sets'][i]][0],idxs[spec_dict['sets'][i]][1]],
                                                      spec_dict['level'], spec_dict['gridsize'])
            
            ax.contour(X, Y, Z, levels=[threshold_value],colors=color, linewidths=spec_dict['linew'])
            legend_elements.append(Line2D([0], [0], color=color, linewidth=spec_dict['linew'],
                                          label=f'{1-spec_dict['level']:.1%}-{spec_dict['sets'][i]}'))

    # Set axis labels and title based on spec_dict
    if 'xlabel' in spec_dict:
        ax.set_xlabel(spec_dict['xlabel'], fontsize=spec_dict['font'] - 2)
    if 'ylabel' in spec_dict:
        ax.set_ylabel(spec_dict['ylabel'], fontsize=spec_dict['font'] - 2)
    if 'title' in spec_dict:
        ax.set_title(spec_dict['title'], fontsize=spec_dict['font'])

    # Set axis limits based on embedding data
    ax.set_xlim(np.min(embedding[:, 0]), np.max(embedding[:, 0]))
    ax.set_ylim(np.min(embedding[:, 1]), np.max(embedding[:, 1]))
    
    # Add the legend to the plot
    ax.legend(handles=legend_elements, loc=spec_dict['loc'], fontsize=spec_dict['font'] - 6)

    return fig, ax

def embedding_split3D(embedding, labels, 
                      spec_dict):
    """
    Visualizes a 3D embedding with scatter plots for selected data subsets 
    (TEST, VAL, and TRAIN).
    The plot includes kernel density estimates (KDE) for each subset, as well as the overlap 
    between the TEST and VAL sets.

    Parameters:
    -----------
    - embedding (np.ndarray): A 2D array of shape (n_samples, 3) containing the 3D embedding points to plot.
    - labels (list of np.ndarray): A list containing three arrays 
    for TEST, VAL, and TRAIN labels, respectively.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    
    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    - ax (matplotlib.axes.Axes): The axes containing the plot.
    - overlap (float): The overlap percentage between the two selected sets (TEST and VAL).
    """

    # Extract label subsets
    labels_test, labels_val, labels_train = labels[0], labels[1], labels[2]
    # Determine the number of samples in each subset
    test_samples  = len(labels_test)
    val_samples   = len(labels_val)
    train_samples = len(labels_train)

    # Index ranges for each subset (TEST, VAL, TRAIN)
    idxs = {'TEST':  [0, len(labels_test)],
            'VAL':   [len(labels_test), len(labels_test) + len(labels_val)],
            'TRAIN': [len(labels_test) + len(labels_val), -1]}
    
    # Create 3D plot
    fig, ax = plt.subplots(figsize=(spec_dict['figdim'][0], spec_dict['figdim'][1]),
                           subplot_kw=dict(projection='3d'),
                           constrained_layout=True)

    probability_thresholds = [(norm.cdf(sigma) - norm.cdf(-sigma)) for sigma in list(spec_dict['sigmalevels'])]

    # Store KDE density values for each dataset
    density_values = []
    
    # Plot data for the first selected set (e.g., TEST or VAL)
    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf
    zmin, zmax = np.inf, -np.inf
    
    for i in range(len(spec_dict['sets'])):
        alpha   = spec_dict['alpha_trainval'] if spec_dict['sets'][i] != 'TEST' else spec_dict['alpha_test']
        size    = spec_dict['s_trainval']     if spec_dict['sets'][i] != 'TEST' else spec_dict['s_test']
        data    = embedding[idxs[spec_dict['sets'][i]][0]:idxs[spec_dict['sets'][i]][1], :]
        values  = data.T
        density = gaussian_kde(values)(values)

        idx = density.argsort()
        sorted_density = np.sort(density[idx])[::-1]
        cdf = np.cumsum(sorted_density)
        cdf /= cdf[-1]  # Normalize to range [0, 1]
        density_thresholds = np.array([sorted_density[np.searchsorted(cdf, level)] for level in probability_thresholds])

        mask = np.array([True]*len(density),dtype=bool)
        if (i==0) and (spec_dict['sigmath'] is not None):
            mask = (density>density_thresholds[spec_dict['sigmalevels']==spec_dict['sigmath']])

        x, y, z, density = data[mask,0], data[mask,1], data[mask,2], density[mask]
        scatter = ax.scatter(x, y, z, c=density, s=size, cmap=spec_dict['cmap'][i],
                             alpha=alpha, label=spec_dict['sets'][i])

        xmin, xmax = np.min([xmin,np.min(x)]), np.max([xmax,np.max(x)])
        ymin, ymax = np.min([ymin,np.min(y)]), np.max([ymax,np.max(y)])
        zmin, zmax = np.min([zmin,np.min(z)]), np.max([zmax,np.max(z)])

        cbar = fig.colorbar(scatter, ax=ax, pad=0.1, fraction=0.03)
        cbar.set_label(spec_dict['sets'][i] + ' set',fontsize=spec_dict['font']-4)
        cbar.set_ticks(density_thresholds)  # Set ticks to sigma levels (1σ, 1.5σ, 2σ, etc.)
        cbar.set_ticklabels([f'{s}$\\sigma$' for s in list(spec_dict['sigmalevels'])])  # Label ticks as sigma values

    # Compute the overlap between the selected sets (TEST and VAL)
    overlap = get_overlap3D(embedding, idxs[spec_dict['sets'][0]], idxs[spec_dict['sets'][1]], (1 - spec_dict['level']), spec_dict['gridsize'])
    
    # Set plot title and axis labels
    ax.set_title(spec_dict['title'], fontsize=spec_dict['font'])
    ax.set_xlabel('Output Activation ' + spec_dict['classlabels'][0], fontsize=spec_dict['font'] - 2)
    ax.set_ylabel('Output Activation ' + spec_dict['classlabels'][1], fontsize=spec_dict['font'] - 2)
    ax.set_zlabel('Output Activation ' + spec_dict['classlabels'][2], fontsize=spec_dict['font'] - 2)

    # Set axis limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    # Remove padding on z-axis label
    ax.zaxis.labelpad = 0  # This can be adjusted based on aesthetics if necessary
    if (spec_dict['view']['elev'] is None) or (spec_dict['view']['azim'] is None) or (spec_dict['view']['roll'] is None):
        ax.view_init(elev=spec_dict['view']['elev'], azim=spec_dict['view']['azim'], roll=spec_dict['view']['roll'])
    
    return fig, ax, overlap

#### EMBEDDING X LABELS: TRAIN-VALIDATION-TEST ####
def embedding_labels2D(embedding, labels, 
                       spec_dict, 
                       ax=None):
    """
    Visualizes a 2D embedding of labeled data points. It includes the following:
    - Scatter plot of data points colored by their labels (optional: based on training or all data).
    - Centroids (baricenters) of each class represented by their labels.
    - Triangular filling between class centroids.
    - KDE plots for density estimation of the classes (optional).
    
    Parameters:
    -----------
    - embedding (np.ndarray): A 2D array of shape (n_samples, 2) containing the 2D embedding points to plot.
    - labels (list of np.ndarray): A list containing three arrays for 
    TEST, VAL, and TRAIN labels, respectively.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - ax (matplotlib.axes.Axes, optional): The axes on which to plot. 
    If None, a new figure and axes are created.
    
    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    - ax (matplotlib.axes.Axes): The axes containing the plot.
    """

    # Extract the labels for each dataset: TEST, VAL, TRAIN
    labels_test, labels_val, labels_train = labels[0], labels[1], labels[2]
    
    # Index ranges for each subset (TEST, VAL, TRAIN)
    idxs = {'TEST':  [0, len(labels_test)],
            'VAL':   [len(labels_test), len(labels_test) + len(labels_val)],
            'TRAIN': [len(labels_test) + len(labels_val), -1]}

    # Concatenate all labels into one array (TEST + VAL + TRAIN)
    labels_tot = labels_test + labels_val + labels_train

    # Calculate mask to extracted the selected sets
    mask1 = np.zeros(len(labels_tot), dtype=bool)
    # Iterate through idxs.keys() and set the mask for corresponding labels
    for t, z in enumerate(idxs.keys()):
        if z in spec_dict['sets']:
            # Get the range of labels for the current set i
            start_idx = sum(len(labels[j]) for j in range(t))
            end_idx = sum(len(labels[j]) for j in range(t + 1))
            # Set the corresponding range of mask1 to True for this set
            mask1[start_idx:end_idx] = True
    
    # Initialize the figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(spec_dict['figdim'][0], spec_dict['figdim'][1]))
    else:
        fig = None

    # Initialize an array to store the centroids (baricenters) of each class
    baricenters = np.zeros((len(spec_dict['classlabels']), 2))
    # Loop through each class label to plot the scatter points
    for i in range(len(spec_dict['classlabels'])):
        
        # Create a boolean mask for the current class label
        mask0 = np.array([element == spec_dict['classlabels'][i] for element in labels_tot], dtype=bool)  
        mask = mask0 & mask1
            
        # Plot the data points for the current class
        ax.scatter(embedding[mask, 0], embedding[mask, 1], color=spec_dict['colors'][i], alpha=spec_dict['alpha'],
                   marker=spec_dict['marker'], s=spec_dict['s'])
        
        # Calculate and store the centroid (mean of coordinates) for each class
        baricenters[i, 0] = np.mean(embedding[mask, 0])
        baricenters[i, 1] = np.mean(embedding[mask, 1])
        
        # Plot the centroid (black point)
        ax.scatter(baricenters[i, 0], baricenters[i, 1], color='k')
        ax.text(baricenters[i, 0], baricenters[i, 1], s=spec_dict['classlabels'][i], fontsize=spec_dict['font'])

    # Fill the area between the class centroids (forming a triangle)
    ax.fill(baricenters[:, 0], baricenters[:, 1], color='k', alpha=spec_dict['alpha_triangle'])

    # Optionally, plot KDE contours if 'dim' is in spec_dict
    if 'dim' in spec_dict:
        df = pd.DataFrame(embedding[mask1, :], columns=['dim0', 'dim1'])
        df['Label'] = np.array(labels_tot)[mask1]
        
        # Generate the KDE plot for class density estimation
        kde_plot = sns.kdeplot(data=df, x=df.columns[0], y=df.columns[1], hue='Label', 
                               hue_order=spec_dict['classlabels'],
                               fill=False, levels=[spec_dict['level']], 
                               palette=spec_dict['colors'], ax=ax)
    
    # Set axis labels and title
    if 'xlabel' in spec_dict:
        ax.set_xlabel(spec_dict['xlabel'], fontsize=spec_dict['font'] - 2)
    if 'ylabel' in spec_dict:
        ax.set_ylabel(spec_dict['ylabel'], fontsize=spec_dict['font'] - 2)
    if 'title' in spec_dict:
        ax.set_title(spec_dict['title'], fontsize=spec_dict['font'])

    # Set axis limits based on the data range
    ax.set_xlim(np.min(embedding[:, 0]), np.max(embedding[:, 0]))
    ax.set_ylim(np.min(embedding[:, 1]), np.max(embedding[:, 1]))
    
    # Create a legend for the plot
    legend_elements = []
    for i in range(len(spec_dict['classlabels'])):
        legend_elements.append(Line2D([0], [0], marker=spec_dict['marker'], color=spec_dict['colors'][i],
                                      label=spec_dict['classlabels'][i], linestyle='None'))

    # Calculate the area of the triangle formed by the centroids
    Area = get_triangle_area(baricenters)
    leg_label = 'Area: ' + str(np.round(Area, 2)) + 'u$^2$'
    legend_elements.append(Line2D([0], [0], marker='^', color='k', label=leg_label, 
                          alpha=spec_dict['alpha_triangle'], linestyle='None'))
    
        # Add the legend to the plot
    ax.legend(handles=legend_elements, loc=spec_dict['loc'], fontsize=spec_dict['font'] - 6,
              title=' & '.join(spec_dict['sets']))
    
    return fig, ax

def embedding_labels3D(embedding, labels, 
                       spec_dict):
    """
    Visualizes a 3D embedding of labeled data points. It includes:
    - Scatter plot of data points colored by their labels (optional: based on training or all data).
    - Centroids (baricenters) of each class represented by their labels.
    - A triangular face formed between the centroids, with area calculated for visualization.
    
    Parameters:
    -----------
    - embedding (np.ndarray): A 2D array of shape (n_samples, 3) containing the 3D embedding points to plot.
    - labels (list of np.ndarray): A list containing three arrays 
    for TEST, VAL, and TRAIN labels, respectively.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    
    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    - ax (matplotlib.axes.Axes): The axes containing the plot. 
    - baricenters (np.ndarray): An array containing the centroids (baricenters) for each class.
    - area (float): The area of the triangle with vertices the baricenters for each class.
    """

    # Extract the labels for each dataset: TEST, VAL, TRAIN
    labels_test, labels_val, labels_train = labels[0], labels[1], labels[2]

    # Index ranges for each subset (TEST, VAL, TRAIN)
    idxs = {'TEST':  [0, len(labels_test)],
            'VAL':   [len(labels_test), len(labels_test) + len(labels_val)],
            'TRAIN': [len(labels_test) + len(labels_val), -1]}
    
    # Concatenate all labels into one array (TEST + VAL + TRAIN)
    labels_tot = labels_test + labels_val + labels_train

    # Calculate mask to extracted the selected sets
    mask1 = np.zeros(len(labels_tot), dtype=bool)
    # Iterate through idxs.keys() and set the mask for corresponding labels
    for t, z in enumerate(idxs.keys()):
        if z in spec_dict['sets']:
            # Get the range of labels for the current set i
            start_idx = sum(len(labels[j]) for j in range(t))
            end_idx = sum(len(labels[j]) for j in range(t + 1))
            # Set the corresponding range of mask1 to True for this set
            mask1[start_idx:end_idx] = True

    # Create a 3D plot
    fig, ax = plt.subplots(figsize=(spec_dict['figdim'][0], spec_dict['figdim'][1]),
                           subplot_kw=dict(projection='3d'))
    
    # Initialize an array to store the centroids (baricenters) of each class
    baricenters = np.zeros((len(spec_dict['classlabels']), 3))
    # Loop through each class label to plot the scatter points
    for i in range(len(spec_dict['classlabels'])):
        
        # Create a boolean mask for the current class label
        mask0 = np.array([element == spec_dict['classlabels'][i] for element in labels_tot], dtype=bool)  
        mask = mask0 & mask1
            
        # Plot the data points for the current class
        ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2], 
                   color=spec_dict['colors'][i], alpha=spec_dict['alpha'], 
                   marker=spec_dict['marker'], s=spec_dict['s'])
        
        # Calculate and store the centroid (mean of coordinates) for each class
        baricenters[i, :] = np.mean(embedding[mask, :], axis=0)
        
        # Plot the centroid (black point) and label it
        ax.scatter(baricenters[i, 0], baricenters[i, 1], baricenters[i, 2], color='k')
        ax.text(baricenters[i, 0], baricenters[i, 1], baricenters[i, 2], 
                s=spec_dict['classlabels'][i], fontsize=spec_dict['font'])

    # Create a Poly3DCollection object with the triangle face formed by the centroids
    poly3d = Poly3DCollection([baricenters], color='k', alpha=spec_dict['alpha_triangle'])
    
    # Add the triangle to the axes
    ax.add_collection3d(poly3d)

    # Set axis labels and title
    ax.set_title(spec_dict['title'], fontsize=spec_dict['font'])
    ax.set_xlabel('Output Activation ' + spec_dict['classlabels'][0], fontsize=spec_dict['font'] - 2)
    ax.set_ylabel('Output Activation ' + spec_dict['classlabels'][1], fontsize=spec_dict['font'] - 2)
    ax.set_zlabel('Output Activation ' + spec_dict['classlabels'][2], fontsize=spec_dict['font'] - 2)
    
    # Create a legend for the plot
    legend_elements = []
    for i in range(len(spec_dict['classlabels'])):
        legend_elements.append(Line2D([0], [0], marker=spec_dict['marker'], color=spec_dict['colors'][i], 
                               label=spec_dict['classlabels'][i], linestyle='None'))

    # Calculate the area of the triangle formed by the centroids
    area = get_triangle_area(baricenters)
    leg_label = 'Area: ' + str(np.round(area, 2)) + 'u$^2$'
    legend_elements.append(Line2D([0], [0], marker='^', color='k', label=leg_label, 
                          alpha=spec_dict['alpha_triangle'], linestyle='None'))
    
    # Add the legend to the plot
    ax.legend(handles=legend_elements, loc=spec_dict['loc'], fontsize=spec_dict['font'] - 6,
              title = ' & '.join(spec_dict['sets']))

    # Set axis limits based on the data range
    ax.set_xlim(np.min(embedding[:, 0]), np.max(embedding[:, 0]))
    ax.set_ylim(np.min(embedding[:, 1]), np.max(embedding[:, 1]))
    ax.set_zlim(np.min(embedding[:, 2]), np.max(embedding[:, 2]))

    # Adjust z-axis label padding
    ax.zaxis.labelpad = 0  # Can be adjusted based on preference
    
    return fig, ax, baricenters, area
    
#### EMBEDDING X CLASSIFICATION MISTAKED ####
def embedding_class2D(embedding, inout_labels, 
                      spec_dict,
                      subj_test=None, nw_test=None, colors_test=None, 
                      ax=None):
    """
    Visualizes a 2D embedding of classification results, highlighting correct and misclassified points.
    Supports 'subjects' mode for subject-based classification and 'classification' mode for general label-based classification.

    Parameters:
    -----------
    - embedding (np.ndarray): Array of shape (n_samples, 2) with 2D coordinates of points to plot.
    - inout_labels (list of lists): A list containing two lists: one for the input labels and 
    one for the output labels.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.
    - subj_test (list, optional): List of subject identifiers (used only if `mode` is 'subjects').
    - nw_test (list, optional): List of sample counts per subject (used only if `mode` is 'subjects').
    - colors_test (list, optional): List of colors per subject (used only if `mode` is 'subjects').
    - ax (matplotlib.axes.Axes, optional): Axes for the plot; if None, creates a new one.

    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    - ax (matplotlib.axes.Axes): The axes containing the plot.
    """

    # Check if the 'ax' (matplotlib axis) object is passed, if not create a new one.
    if ax is None:
        # Create a figure and axis using the specified figure dimensions in 'spec_dict'
        fig, ax = plt.subplots(1, 1, figsize=(spec_dict['figdim'][0], spec_dict['figdim'][1]))
    else:
        fig = None  # No need to create a new figure if 'ax' is provided
    
    # Determine the number of test samples based on the length of the first element in 'inout_labels'
    test_samples = len(inout_labels[0])

    # Scatter plot for validation and training data 
    ax.scatter(embedding[test_samples:, 0], embedding[test_samples:, 1], color='k', 
               alpha=spec_dict['alpha_trainval'], marker=spec_dict['marker'], s=spec_dict['s_trainval'])  # VAL + TRAINING
    
    legend1_elements = [Line2D([0], [0], alpha=spec_dict['alpha_trainval'], marker=spec_dict['marker'],
                               markersize=spec_dict['s_trainval'],color='k', label='TRAIN+VAL', linestyle='None')]
                        
    # Separate the test embedding data
    embedding_test = embedding[:test_samples, :]
    # Create a mask where the true labels are equal to the predicted labels
    mask = (np.array(inout_labels[0]) == np.array(inout_labels[1]))
    
    # If the mode is 'subjects', we plot based on subject-specific information
    if spec_dict['mode'] == 'subjects':
        if (nw_test is not None) and (colors_test is not None) and (subj_test is not None):
            # PLOT TEST
            embedding_test = embedding[:test_samples, :]
            index = 0  # Start index for the subjects
            legend_elements = []  # List to store the legend elements
    
            # Loop through each subject and plot their data
            for i, samples in enumerate(nw_test):
                current_mask = mask[index:(index + samples)]  # Mask for the current subject's data
    
                # Select correct and incorrect classifications for the current subject
                correct = embedding_test[index:(index + samples), :][current_mask, :]
                not_correct = embedding_test[index:(index + samples), :][~current_mask, :]
    
                # Plot correct classifications (dots) and incorrect ones (pluses) using different colors
                ax.scatter(correct[:, 0], correct[:, 1], color=colors_test[index],
                           alpha=spec_dict['alpha_test'], marker='.', s=spec_dict['s_test'])
                ax.scatter(not_correct[:, 0], not_correct[:, 1], color=colors_test[index],
                           alpha=spec_dict['alpha_test'], marker='+', s=spec_dict['s_test'])
    
                # Compute 2D KDE and plot the contour line at the threshold value
                X, Y, Z, threshold_value = compute_2D_KDE(embedding, [index, (index + samples)], 
                                                          spec_dict['level'], spec_dict['gridsize'])
                contour = ax.contour(X, Y, Z, levels=[threshold_value], colors=[colors_test[index]], linewidths=spec_dict['linew'])
    
                # Create a label for the legend with subject information
                max_subj_length = max(len(str(s)) for s in subj_test)  
                leg_label = f'S{subj_test[i]:0>{int(np.log10(spec_dict['nsub']))+1}} ({samples:<{int(np.log10(samples))+1}}) | {sum(current_mask)/samples:>2.0%}'
                legend_elements.append(Line2D([0], [0], marker=spec_dict['marker'], color=colors_test[index], 
                                              label=leg_label, linestyle='None'))
    
                # Update the index for the next subject
                index += nw_test[i]
    
            # Add the subject-specific legend to the plot
            subject_legend = ax.legend(handles=legend_elements, loc=spec_dict['loc_subj'], fontsize=spec_dict['font'] - 6)
            ax.add_artist(subject_legend)
        else:
            # Raise an error if any of the necessary inputs (nw_test, colors_test, or subj_test) is missing
            raise ValueError("One between nw_test, colors_test, and subj_test is None.")
    
        # Create the legend for correct vs misclassified points in black color
        legend1_elements.append(Line2D([0], [0], marker='.', color='k', label=f'Correct Classified: {sum(mask)}', linestyle='None'))
        legend1_elements.append(Line2D([0], [0], marker='+', color='k', label=f'Misclassified: {sum(~mask)}', linestyle='None'))
    
    # If the mode is 'classification', we plot based on classification results
    elif spec_dict['mode'] == 'classification':
        # Plot the correct classifications in green (dots)
        ax.scatter(embedding_test[mask, 0], embedding_test[mask, 1],
                   color='g', alpha=spec_dict['alpha_test'], marker='.', s=spec_dict['s_test'])
    
        # Compute 2D KDE for correct classifications and plot the contour
        X, Y, Z, threshold_value = compute_2D_KDE(embedding_test[mask, :], [0, -1], spec_dict['level'], spec_dict['gridsize'])
        contour = ax.contour(X, Y, Z, levels=[threshold_value], colors=['g'], linewidths=spec_dict['linew'])
    
        # Plot the misclassified points in red (pluses)
        ax.scatter(embedding_test[~mask, 0], embedding_test[~mask, 1], 
                   color='r', alpha=spec_dict['alpha_test'], marker='+', s=spec_dict['s_test'])
    
        # Compute 2D KDE for misclassified points and plot the contour
        X, Y, Z, threshold_value = compute_2D_KDE(embedding_test[~mask, :], [0, -1], spec_dict['level'], spec_dict['gridsize'])
        contour = ax.contour(X, Y, Z, levels=[threshold_value], colors=['r'], linewidths=spec_dict['linew'])
    
        # Create the legend for correct vs misclassified points in red and green colors
        legend1_elements.append(Line2D([0], [0], marker='.', color='g', label=f'Correct Classified: {sum(mask)}', linestyle='None'))
        legend1_elements.append(Line2D([0], [0], marker='+', color='r', label=f'Misclassified: {sum(~mask)}', linestyle='None'))
    
    # Add xlabel, ylabel, and title if specified in the 'spec_dict'
    if 'xlabel' in spec_dict:
        ax.set_xlabel(spec_dict['xlabel'], fontsize=spec_dict['font'] - 2)
    if 'ylabel' in spec_dict:
        ax.set_ylabel(spec_dict['ylabel'], fontsize=spec_dict['font'] - 2)
    if 'title' in spec_dict:
        ax.set_title(spec_dict['title'], fontsize=spec_dict['font'])
    
    # Set the limits of the x and y axes to the min and max values of the embedding
    ax.set_xlim(np.min(embedding[:, 0]), np.max(embedding[:, 0]))
    ax.set_ylim(np.min(embedding[:, 1]), np.max(embedding[:, 1]))
    
    # Add the final legend to the plot (for correct vs misclassified points)
    ax.legend(handles=legend1_elements, loc=spec_dict['loc'], fontsize=spec_dict['font'] - 6)
    
    # Return the figure and axis objects
    return fig, ax

###################################
####    EMBEDDING UTILS FUNC    ###
###################################
def compute_2D_KDE(embedding, set_index, prob=0.6827, grid_size=100j):
    """
    Computes a 2D Kernel Density Estimate (KDE) for the given subset of data points in the 2D embedding.
    Returns the KDE values over a grid, the threshold corresponding to the specified cumulative probability,
    and the grid coordinates.

    Parameters:
    -----------
    - embedding (np.ndarray): A 2D array of shape (n_samples, 2) containing 2D data points.
    - set_index (tuple): A tuple (start, end) to specify the slice of the data to use for KDE computation.
    - prob (float): The cumulative probability threshold to define the KDE level.
    Default is 0.6827 (approximately the 1-sigma region for a normal distribution).
    - grid_size (complex, optional): The resolution of the grid used for KDE evaluation. 
    Default is `100j`, which corresponds to 100 grid points along each axis.

    Returns:
    --------
    - X (np.ndarray): The X grid coordinates over which the KDE is evaluated.
    - Y (np.ndarray): The Y grid coordinates over which the KDE is evaluated.
    - kde_values (np.ndarray): The KDE values on the 2D grid, spanned by (X,Y).
    - threshold_value (float): The KDE density value corresponding to the specified cumulative probability.
    """
    
    # Extract the set1 data points and evaluate their KDE densities
    data = embedding[set_index[0]:set_index[1], :]
    kde  = gaussian_kde(data.T)
    
    # Define the grid limits based on the data's min and max values for each dimension
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    
    # Create the 2D grid on which to evaluate the KDE
    x, y = np.mgrid[x_min:x_max:grid_size, y_min:y_max:grid_size]
    
    # Flatten the grid coordinates to evaluate KDE at each point
    positions  = np.vstack([x.ravel(), y.ravel()])
    
    # Evaluate the KDE over the grid
    kde_values = kde(positions).reshape(x.shape)
    
    # Sort the KDE values to calculate the cumulative distribution function (CDF)
    sorted_kde_values = np.sort(kde_values, axis=None)#[::-1]
    cdf = np.cumsum(sorted_kde_values)
    cdf /= cdf[-1]  # Normalize CDF to range [0, 1]
    
    # Find the threshold value corresponding to the target cumulative probability level
    threshold_idx = np.searchsorted(cdf, prob)
    threshold     = sorted_kde_values[threshold_idx]
    
    # Return the grid coordinates, KDE values, and the threshold value
    return x, y, kde_values, threshold
    
def get_overlap3D(embedding, set1_index, set2_index, prob=0.6827, grid_size=100j):
    """
    Computes the overlap between two sets of 3D data points based on their probability density 
    estimated using Kernel Density Estimation (KDE).

    Parameters:
    -----------
    - embedding (np.ndarray): A 2D array of shape (n_samples, 3) containing the 3D data points.
    - train_index (tuple): A tuple of two integers (start, end) indicating the slice of the embedding 
    that corresponds to the set1.
    - test_index (tuple): A tuple of two integers (start, end) indicating the slice of the embedding 
    that corresponds to the set2.
    - prob (float, optional): The probability threshold to use for determining overlap. 
    Default is 0.6827 (approximately the 1-sigma region for a normal distribution).
    - grid_size (complex, optional): The resolution of the grid used for KDE evaluation. 
    Default is `100j`, which corresponds to 100 grid points along each axis.

    Returns:
    --------
    - overlap (float): The fraction of set2 points that lie within the KDE threshold
    of the set1 data's density.
    """
    
    # Extract the set1 data points and evaluate their KDE densities
    data = embedding[set1_index[0]:set1_index[1], :]
    kde  = gaussian_kde(data.T)
    
    # Define the grid limits based on the data's min and max values for each dimension
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    z_min, z_max = np.min(data[:, 2]), np.max(data[:, 2])
    
    # Create the 3D grid on which to evaluate the KDE
    x, y, z = np.mgrid[x_min:x_max:grid_size, y_min:y_max:grid_size, z_min:z_max:grid_size]
    
    # Flatten the grid coordinates to evaluate KDE at each point
    positions  = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    
    # Evaluate the KDE over the grid
    kde_values = kde(positions).reshape(x.shape)
    
    # Find the 1-sigma threshold corresponding to the specified probability
    sorted_kde_values = np.sort(kde_values, axis=None)#[::-1]
    cdf = np.cumsum(sorted_kde_values)
    cdf /= cdf[-1]  # Normalize CDF to range [0, 1]
    
    # Find the density threshold corresponding to the cumulative probability
    threshold_idx = np.searchsorted(cdf, prob)
    threshold     = sorted_kde_values[threshold_idx]
    
    # Extract the set2 data points and evaluate their KDE densities
    external_points    = embedding[set2_index[0]:set2_index[1], :]
    external_densities = kde(external_points.T)
    
    # Determine which set2 points have densities exceeding the threshold got from the set1 points
    list_check = (external_densities >= threshold)
    # Compute the overall overlap as the fraction of set2 points above the threshold
    overlap = np.mean(list_check)

    return overlap

def get_triangle_area(vertices):
    """
    Calculates the area of a triangle in 2D or 3D space given its vertices.

    Parameters:
    -----------
    - vertices (np.ndarray): An array containing the coordinates of the triangle's vertices.
      For a 2D triangle, this should be a (3, 2) array.
      For a 3D triangle, this should be a (3, 3) array.
      Each row corresponds to a vertex.

    Returns:
    --------
    - area (float): The area of the triangle.
    """
    # Ensure input is a numpy array
    vertices = np.asarray(vertices)
    
    # Check the shape of the vertices to determine 2D or 3D
    if vertices.shape[1] == 2:  # 2D case
        # Extract vertices
        A, B, C = vertices
        # Calculate the area using the determinant formula
        area = 0.5 * np.abs(A[0] * (B[1] - C[1]) +
                            B[0] * (C[1] - A[1]) +
                            C[0] * (A[1] - B[1]))
    elif vertices.shape[1] == 3:  # 3D case
        # Extract vertices
        A, B, C = vertices
        # Calculate vectors AB and AC
        AB = B - A
        AC = C - A
        # Calculate the cross product of AB and AC
        cross_product = np.cross(AB, AC)
        # Area is half the magnitude of the cross product
        area = np.linalg.norm(cross_product) / 2
    else:
        raise ValueError("Vertices should be a (3, 2) array for 2D or a (3, 3) array for 3D.")

    return area

def get_area3D(embedding, set_index, classlabels, labels):
    """
    Calculates the area of a triangle in 3D space formed by the centroids of specified classes.

    This function takes embedding data, extracts a subset based on `set_index`, computes the centroids 
    (barycenters) of the specified classes (`classlabels`), and calculates the area of the triangle 
    formed by these centroids in 3D space.

    Parameters:
    -----------
    - embedding (np.ndarray): A 2D array of shape (n_samples, n_features) representing the data points.
    - set_index (tuple of int): A tuple (start_index, end_index) specifying the range of rows in the embedding to process.
    - classlabels (list of str): A list of unique class labels for which centroids are computed.
    - labels (list of lists): A list containing three sublists: labels for TEST, VAL, and TRAIN datasets.

    Returns:
    --------
    - Area (float): The area of the triangle formed by the centroids of the specified classes.
    """
    
    data   = embedding[set_index[0]:set_index[1], :]

    # Concatenate all labels into one array (TEST + VAL + TRAIN)
    labels_tot = np.concatenate(labels)
    labels = labels_tot[set_index[0]:set_index[1]]
    
    # Initialize an array to store the centroids (baricenters) of each class
    baricenters = np.zeros((len(classlabels), np.shape(data)[1]))
    
    # Loop through each class label to plot the scatter points
    for i in range(len(classlabels)):
        
        # Create a boolean mask for the current class label
        mask0 = np.array([element == classlabels[i] for element in labels], dtype=bool)
        
        # Calculate and store the centroid (mean of coordinates) for each class
        baricenters[i, :] = np.mean(data[mask0, :], axis=0)
        
    # Calculate the area of the triangle formed by the centroids
    Area = get_triangle_area(baricenters)
    
    return Area

###################################
####      OTHER UTILS FUNC      ###
###################################
def quartile_coefficient_of_variation(data):
    """
    Calculate the Quartile Coefficient of Variation (QCV) for a given dataset.

    The Quartile Coefficient of Variation is a measure of relative spread,
    specifically the ratio of the interquartile range (IQR) to the sum of the
    first and third quartiles. It gives a sense of how spread out the middle 50%
    of the data is relative to its central values.

    Parameters:
    -----------
    - data (np.array): An array for which the QCV is computed.

    Returns:
    --------
    - qcv (float): The Quartile Coefficient of Variation for the given array.
    """
    
    # Calculate the first quartile (Q1) or 25th percentile
    Q1 = np.percentile(data, 25)
    # Calculate the third quartile (Q3) or 75th percentile
    Q3 = np.percentile(data, 75)
    
    # Calculate the Quartile Coefficient of Variation as (Q3 - Q1) / (Q3 + Q1)
    # This ratio represents the spread between the quartiles relative to the center
    qcv = (Q3 - Q1) / (Q3 + Q1)
    
    return qcv

def adjusted_R2(r_value, n, p):
    """
    Calculate the adjusted R-squared (R²) value.

    Adjusted R² is a modified version of R² that accounts for the number of predictors in the model. 
    It provides a more accurate measure of model fit by penalizing for the addition of unnecessary predictors, 
    preventing overfitting and offering a more reliable measure when comparing models with different numbers of predictors.

    Parameters:
    ----------
    - r_value (numpy.array): The R-squared (R²) value of the model.
    - n (int): The total number of observations in the dataset.
    - p (int): The number of predictors (independent variables) in the model.

    Returns:
    -------
    - adj_rvalue (numpy.array): The adjusted R-squared value, accounting for the number of predictors and observations.

    Formula:
    -------
    Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
    """
    # Calculate adjusted R² by incorporating number of observations (n) and predictors (p)
    adj_rvalue = 1 - (1 - r_value**2) * (n - 1) / (n - p - 1)
    return adj_rvalue

###################################
#### OVER-FITTING: LOSS CURVES  ###
###################################
def loss_curve(ax, acc, lims, spec_dict):
    """
    Plots loss curves for given accuracy data on a provided Matplotlib axis.

    Parameters:
    -----------
    - ax (matplotlib.axes.Axes): The axis on which to plot the loss curves.
    - acc (list of list of float): Each sublist contains loss values across epochs for a particular run or split.
    - lims (list of lists): Contains two sub-lists specifying limits for x-axis (epochs) and y-axis (loss).
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.

    Returns:
    --------
    - None: The function modifies the provided axis `ax`.
    """
    
    # Set x-axis (epochs) and y-axis (loss) limits based on input limits and acc data
    xlim, ylim = lims[0].copy(), lims[1].copy()  # Deep copies to avoid modifying the original limits

    # Determine epoch lengths and loss ranges across runs
    epoch_lengths = [len(run) for run in acc]
    max_losses = [np.max(run) for run in acc]
    min_losses = [np.min(run) for run in acc]

    # Automatically adjust xlim and ylim if they are not specified
    if xlim[1] is None:
        xlim[1] = np.max(epoch_lengths)
    if xlim[0] is None:
        xlim[0] = np.min(epoch_lengths)
    if ylim[1] is None:
        ylim[1] = np.max(max_losses)
    if ylim[0] is None:
        ylim[0] = np.min(min_losses)

    # Plot each loss curve with specified color and line width from spec_dict
    for run_loss in acc:
        ax.plot(np.arange(1, len(run_loss) + 1), run_loss, color=spec_dict['color'], linewidth=spec_dict['linew'])
    
    # Set plot title and axis labels with specified font sizes
    ax.set_title(spec_dict['title'], fontsize=spec_dict['font'])
    ax.set_xlabel('Epochs', fontsize=spec_dict['font'] - 2)
    ax.set_ylabel('Loss', fontsize=spec_dict['font'] - 2)
    
    # Apply axis limits and set logarithmic scale for the x-axis (epochs)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xscale('log')

    # Hide x-axis labels and ticks if xlabel is set to False
    if not spec_dict['xlabel']:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel('')
    
    # Set tick parameters with specified font size adjustments
    ax.tick_params(axis='both', which='major', labelsize=spec_dict['font'] - 4)

    # Set the number of y-ticks and format y-tick labels
    ax.set_yticks(np.linspace(ylim[0], ylim[1], spec_dict['ytick']))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))

def loss_correlation(ax, acc_train, acc_val, lims, spec_dict):
    """
    Plots a histogram of the Pearson correlation coefficients between training and validation accuracy for multiple runs.

    Parameters:
    -----------
    - ax (matplotlib.axes.Axes): The axis on which to plot the histogram.
    - acc_train (list of list of float): Training accuracy values for each run.
    - acc_val (list of list of float): Validation accuracy values for each run.
    - lims (list of lists): Contains two sub-lists specifying x-axis and y-axis limits.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.

    Returns:
    --------
    - res_v (list of float): Pearson correlation coefficients between training and validation accuracy for each run.
    """
    
    # Set x-axis (correlation) and y-axis (frequency) limits
    xlim, ylim = lims[0].copy(), lims[1].copy()

    # Calculate Pearson correlation coefficient for each run and store results
    res_v = [pearsonr(acc_train[i], acc_val[i]).statistic for i in range(len(acc_train))]

    # Plot histogram of Pearson correlation values
    ax.hist(res_v, bins=np.arange(xlim[0], xlim[1] + spec_dict['bin'], spec_dict['bin']),
            edgecolor='black', color=spec_dict['color'], linewidth=spec_dict['linew'])
    
    # Configure x and y tick intervals and formatting
    ax.set_xticks(np.arange(xlim[0], xlim[1] + spec_dict['xtick'], spec_dict['xtick']))
    ax.set_yticks(np.linspace(ylim[0], ylim[1], spec_dict['ytick']))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.tick_params(axis='x', labelsize=spec_dict['font'] - 4, rotation=spec_dict['rotation'])
    ax.tick_params(axis='y', labelsize=spec_dict['font'] - 4)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    # Set axis labels and title with specified font sizes
    ax.set_ylabel('$N^{\\circ}$ splits', fontsize=spec_dict['font'] - 2)
    ax.set_xlabel('Pearson $\\rho$', fontsize=spec_dict['font'] - 2)
    ax.set_title(spec_dict['title'], fontsize=spec_dict['font'])
    
    # Add vertical line at zero for reference
    ax.axvline(0, linewidth=spec_dict['linew'], linestyle='--', color='black')

    # Hide x-axis labels if specified
    if not spec_dict['xlabel']:
        ax.set_xticklabels([])
        ax.set_xlabel('')  # Disable x-label
    
    return res_v

def overfitting_inspection(models, acc_train, acc_val, spec_dict):
    """
    Dynamically creates a subplot mosaic to inspect overfitting across multiple models by plotting
    loss curves and loss correlations.

    Parameters:
    -----------
    - models (list of str): Model names to be plotted.
    - acc_train (dict of list of list of float): Dictionary where keys are model names and values 
      are lists containing training accuracy data for each model.
    - acc_val (dict of list of list of float): Dictionary where keys are model names and values 
      are lists containing validation accuracy data for each model.
    - spec_dict (dict): A dictionary containing various specification parameters 
    and other plotting configurations.

    Returns:
    --------
    - fig (matplotlib.figure.Figure): The figure containing the plots.
    """
    
    # Define the subplot layout based on the number of models
    fig, axd = plt.subplot_mosaic(
        [['left'] + [f'center_{i}'] + [f'right_{i}'] for i in range(len(models))],
        figsize=(spec_dict['figdim'][0], spec_dict['figdim'][1]),
        constrained_layout=True
    )
    
    # Initialize legend and set label display options
    spec_dict['xlabel'] = True
    legend_elements = []
    t=0
    for model, color in zip(models, spec_dict['listcolors']):
        spec_dict['title'] = spec_dict['titlet']
        spec_dict['color'] = color
        loss_curve(axd['left'], acc_train[model], [spec_dict['xlimt'], spec_dict['ylimt']], spec_dict)
        legend_elements.append(Line2D([0], [0], color=color, label=spec_dict['models'][t]))
        t+=1
    axd['left'].legend(handles=legend_elements, loc=spec_dict['loc'], fontsize=spec_dict['font'] - 6)
    
    # Plot validation curves and correlation histograms
    for idx, (model, color) in enumerate(zip(models, spec_dict['listcolors'])):
        ax_pos = f'center_{idx}'
        spec_dict.update({'color': color, 'xlabel': idx == len(models) - 1, 'title': '' if idx != 0 else spec_dict['titlev']})
        loss_curve(axd[ax_pos], acc_val[model], [spec_dict['xlimv'], spec_dict['ylimv']], spec_dict)
        
        ax_pos = f'right_{idx}'
        spec_dict.update({'title': '' if idx != 0 else spec_dict['titlec']})
        _ = loss_correlation(axd[ax_pos], acc_train[model], acc_val[model], [spec_dict['xlimc'], spec_dict['ylimc']], spec_dict)
    
    return fig
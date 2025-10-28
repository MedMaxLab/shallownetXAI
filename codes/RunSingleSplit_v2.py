# ===========================
#  Section 1: package import
# ===========================
# This section includes all the packages to import. 
# To run this notebook, you must install in your environment. 
# They are: numpy, pandas, matplotlib, scipy, scikit-learn, pytorch, selfeeg

import argparse
import glob
from itertools import chain, combinations, product
import math
import os
import random
import pickle
import copy
import warnings
warnings.filterwarnings(
    "ignore", message = "Using padding='same'", category = UserWarning
)

# IMPORT STANDARD PACKAGES
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import welch, firwin

# IMPORT TORCH
import torch
import torch.nn as nn
from torchaudio import transforms
from torch.utils.data import DataLoader

# IMPORT SELFEEG 
import selfeeg
import selfeeg.models as zoo
import selfeeg.dataloading as dl

# IMPORT REPOSITORY FUNCTIONS
import AllFnc
from AllFnc import split
from AllFnc.models import (
    ShallowNet,
    ShallowNet2,
    ShallowNetEncoder2,
)
from AllFnc.training import (
    loadEEG,
    lossBinary,
    lossMulti,
    train_model,
    get_performances,
    GetLearningRate,
)
from AllFnc.utilities import (
    restricted_float, positive_float, positive_int_nozero, positive_int, str2bool
)
from AllFnc import eegvislib

if __name__ == '__main__':
    # ===========================
    #  Section 2: set parameters
    # ===========================
    # In this section all tunable parameters are instantiated. The entire training 
    # pipeline is configured here, from the task definition to the model evaluation.
    # Other code cells compute their operations using the given configuration. 
    
    help_d = """
    RunKfold run a single training with a specific split 
    extracted from a nested k-fold subject split 
    (10 outer folds, 5 inner folds). Many parameters 
    can be set, which will be then used to create a custom 
    file name. The only one required is the root dataset path.
    Others have a default in case you want to check a single demo run.
    
    Example:
    
    $ python RunKfold -d /path/to/data
    
    """
    parser = argparse.ArgumentParser(description=help_d)
    parser.add_argument(
        "-d",
        "--datapath",
        dest      = "dataPath",
        metavar   = "datasets path",
        type      = str,
        nargs     = 1,
        required  = True,
        help      = """
        The dataset path. This is expected to be static across all trainings. 
        dataPath must point to a directory which contains four subdirecotries, one with 
        all the pickle files containing EEGs preprocessed with a specific pipeline.
        Subdirectoties are expected to have the following names, which are the same as
        the preprocessing pipelinea to evaluate: 1) raw; 2) filt; 3) ica; 4) icasr
        """,
    )
    parser.add_argument(
        "-p",
        "--pipeline",
        dest      = "pipelineToEval",
        metavar   = "preprocessing pipeline",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'filt',
        choices   =['raw', 'filt', 'ica', 'icasr'],
        help      = """
        The pipeline to consider. It can be one of the following:
        1) raw; 2) filt; 3) ica; 4) icasr
        """,
    )
    parser.add_argument(
        "-t",
        "--task",
        dest      = "taskToEval",
        metavar   = "task",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'eyes',
        choices   =['eyes', 'parkinson', 'motorimagery',
                    'alzheimer', 'alzheimerca', 'alzheimercf', 'alzheimeraf',
                    'sleep', 'cognitive'],
        help      = """
        The task to evaluate. It can be one of the following:
        1) eyes; 3) parkinson; 4) motorimagery; 5) sleep; 6) cognitive;
        2) alzheimer; 7) alzheimerca; 8) alzheimercf; 9) alzheimeraf
        """,
    )
    parser.add_argument(
        "-m",
        "--model",
        dest      = "modelToEval",
        metavar   = "model",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'shallownet',
        choices   = ["eegnet", "shallownet", "deepconvnet", "fbcnet", "atcnet", "eegconf","xeegnet"],
        help      = """
        The model to evaluate. It can be one of the following:
        1) eegnet; 2) shallownet; 3) deepconvnet; 4) resnet; 
        5) eegsym; 6) atcnet; 7) hybridnet; 8) psdnet
        """,
    )
    parser.add_argument(
        "-f",
        "--outer",
        dest      = "outerFold",
        metavar   = "outer fold",
        type      = int,
        nargs     = '?',
        required  = False,
        default   = 1,
        choices   = range(1,11),
        help      = 'The outer fold to evaluate. It can be a number between 1 and 10'
    )
    parser.add_argument(
        "-i",
        "--inner",
        dest      = "innerFold",
        metavar   = "inner fold",
        type      = int,
        nargs     = '?',
        required  = False,
        default   = 1,
        choices   = range(1,6),
        help      = 'The inner fold to evaluate. It can be a number between 1 and 5'
    )
    parser.add_argument(
        "-s",
        "--downsample",
        dest      = "downsample",
        metavar   = "downsample",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = True,
        help      = """
        A boolean that set if downsampling at 125 Hz should be applied or not.
        The presented analysis uses 250 Hz, which is 5.55 times the maximum investigated 
        frequency (45 Hz). Note that models usually perform better with 125 Hz. 
        For example, EEGnet was tuned on 128 Hz.
        """
    )
    parser.add_argument(
        "-z",
        "--zscore",
        dest      = "z_score", 
        metavar   = "zscore",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = True,
        help      = """
        A boolean that set if the z-score should be applied or not. 
        The presented analysis applied the z-score, as different preprocessing pipelines
        produce EEGs that evolve on different range of values.
        """
    )
    parser.add_argument(
        "-r",
        "--rminterp",
        dest      = "rem_interp",
        metavar   = "remove interpolated",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = False,
        help      = """
        A boolean that set if the interpolated channels should be 
        removed or not. BIDSAlign aligns all EEGs to a common 61 channel template based
        on the 10_10 International System.
        """
    )
    parser.add_argument(
        "-b",
        "--batch",
        dest      = "batchsize",
        metavar   = "batch size",
        type      = positive_int_nozero,
        nargs     = '?',
        required  = False,
        default   = 64,
        help      = """
        Define the Batch size. It is suggested to use 64 or 128.
        The experimental analysis was performed on batch 64.
        """
    )
    parser.add_argument(
        "-o",
        "--overlap",
        dest      = "overlap",
        metavar   = "windows overlap",
        type      = restricted_float,
        nargs     = '?',
        required  = False,
        default   = 0.0,
        help      = """
        The overlap between time windows. Higher values means more samples 
        but higher correlation between them. 0.25 is a good trade-off.
        Must be a value in [0,1)
        """
    )
    parser.add_argument(
        "-l",
        "--learningrate",
        dest      = "lr",
        metavar   = "learning rate",
        type      = positive_float,
        nargs     = '?',
        required  = False,
        default   = 0.0,
        help      = """
        The learning rate. If left to its default (zero) a proper learning rate
        will be chosen depending on the model and task to evaluate. Optimal learning
        rates were identified by running multiple trainings with different set of values.
        Must be a positive value
        """
    )
    parser.add_argument(
        "-c",
        "--cwindow",
        dest      = "window",
        metavar   = "window",
        type      = positive_float,
        nargs     = '?',
        required  = False,
        default   = 4.0,
        help      = """
        The window (input) size, in seconds. Each EEG will be partitioned in
        windows of length equals to the one specified by this input.
        c was the first available letter.
        """
    )
    parser.add_argument(
        "-j",
        "--subject",
        dest      = "subject",
        metavar   = "subject",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = True,
        help      = """
        A boolean that set if the model should be trained with a subject invariant
        method. This actually adds a network new network head used to predict
        the subject id. Then, correct id predictions will be penalized so to avoid
        the creation of features highly dominated by subject-specific
        characteristics.
        """
    )
    parser.add_argument(
        "-a",
        "--appleloss",
        dest      = "appleloss",
        metavar   = "appleloss",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = True,
        help      = """
        A boolean that set if the subject loss to use must be a fake id
        cross entropy or the cross entropy variant proposed by Apple in the
        paper "Subject-aware contrastive learning for biosignals
        """
    )
    parser.add_argument(
        "-w",
        "--workers",
        dest      = "workers",
        metavar   = "dataloader workers",
        type      = positive_int,
        nargs     = '?',
        required  = False,
        default   = 0,
        help      = """
        The number of workers to set for the dataloader. Datasets are preloaded
        for faster computation, so 0 is preferred due to known issues on values
        greater than 1 for some os, and to not increase too much the memory usage.
        """
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest      = "verbose",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = False,
        help      = """
        Set the verbosity level of the whole script. If True, information about
        the choosen split, and the training progression will be displayed
        """
    )
    args = vars(parser.parse_args())
    
    if args['verbose']:
        print('running training with the following parameters:')
        print(' ')
        for key in args:
            if key == 'dataPath':
                print( f"{key:15} ==> {args[key][0]:<15}") 
            else:
                print( f"{key:15} ==> {args[key]:<15}") 
    
    dataPath       = args['dataPath'][0]
    pipelineToEval = args['pipelineToEval']
    taskToEval     = args['taskToEval']
    modelToEval    = args['modelToEval']
    outerFold      = args['outerFold'] - 1
    innerFold      = args['innerFold'] - 1
    downsample     = args['downsample']
    z_score        = args['z_score']
    rem_interp     = args['rem_interp']
    batchsize      = args['batchsize']
    overlap        = args['overlap']
    workers        = args['workers']
    window         = args['window']
    verbose        = args['verbose']
    subjectTrain   = args['subject']
    appleLoss      = args['appleloss']
    lr             = args['lr']
    seed           = 83136297

    # Define the device to use
    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")
    
    # fold to eval is the correct index to get the desired train/val/test partition
    foldToEval = outerFold*5 + innerFold
    
    # ==================================
    #  Section 3: create partition list
    # ==================================
    # To create the partition list we will use two functions:
    # 1) create_nested_kfold_subject_split, which creates a list where each index 
    #    include three lists, the first with the subject's IDs to put in the training 
    #    set, the second with the subject's IDs to put in the validation set, the third
    #    with the remaining subjects included in the test set. Since it is a nested 
    #    k-fold subject-based split 10 outer folds will be created 
    #    (total subject --> train/test split), then for each outer fold 5 inner splits
    #    will be created (train --> train/validation split).
    # 2) merge_partition_lists, which merges two lists into a final partition list. To 
    #    create stratified splits, create_nested_kfold_subject_split is called for each
    #    category. Then, label-specific lists are merged.
    
    if 'alzheimer' in taskToEval.casefold():
        # ALZ = subjects 1 to 36; CTL = subjects 37 to 65; FTD = subjects 66 to 88
        a_id = [i for i in range(1,37)]
        c_id = [i for i in range(37,66)]
        f_id = [i for i in range(66,89)]
        
        part_a = split.create_nested_kfold_subject_split(a_id, 10, 5)
        part_c = split.create_nested_kfold_subject_split(c_id, 10, 5)
        part_f = split.create_nested_kfold_subject_split(f_id, 10, 5)
    
        if taskToEval.casefold() == 'alzheimer':
            partition_list_1 = split.merge_partition_lists(part_a, part_c, 10, 5)
            partition_list = split.merge_partition_lists(partition_list_1, part_f, 10, 5)
        elif taskToEval.casefold() == 'alzheimerca':
            partition_list = split.merge_partition_lists(part_a, part_c, 10, 5)
        elif taskToEval.casefold() == 'alzheimercf':
            partition_list = split.merge_partition_lists(part_c, part_f, 10, 5)
        elif taskToEval.casefold() == 'alzheimeraf':
            partition_list = split.merge_partition_lists(part_a, part_f, 10, 5)
    
    elif taskToEval.casefold() == 'cognitive':
        # CTL = subjects 101 to 149; PD/PDD/PDMCI = mixing number in [1; 100]
        c_id = [i for i in range(101,150)]
        pd_id = [3, 6, 7, 16, 17, 18, 21, 24, 26, 27, 30, 32, 35, 37, 39, 40, 45,
                 46, 50, 51, 53, 57, 58, 60, 61, 62, 63, 65, 67, 68, 69, 71, 74,
                 76, 80, 81, 82, 83, 84, 85, 86, 87, 90, 92, 93, 94, 100]
        pdd_id = [1, 2, 5, 8, 9, 12, 13, 15, 22, 25, 33, 38, 44, 48, 75, 78, 88, 95, 96]
        pdmci_id = [4, 10, 11, 14, 19, 20, 23, 28, 29, 31, 34, 36, 41, 42, 43, 47,
                    49, 52, 54, 55, 56, 59, 64, 66, 70, 72, 73, 77, 79, 89, 91, 97,
                    98, 99]
        part_c = split.create_nested_kfold_subject_split(c_id, 10, 5)
        part_p = split.create_nested_kfold_subject_split(pd_id, 10, 5)
        part_d = split.create_nested_kfold_subject_split(pdd_id, 10, 5)
        part_m = split.create_nested_kfold_subject_split(pdmci_id, 10, 5)
        
        # first --> mix two groups; then --> mix the mix
        # splits have a more similar number of subject per set in this way
        partition_1 = split.merge_partition_lists(part_c, part_m, 10, 5)
        partition_2 = split.merge_partition_lists(part_p, part_d, 10, 5)
        partition_list = split.merge_partition_lists(partition_1, partition_2, 10, 5)
        
    # ======================================
    # Section 4: set the training parameters
    # =====================================
    
    # This section sets other parameters necessary to start the training pipeline. 
    # Such parameters are necessary to:
    # customize the EEG loading function.
    # define the Pytorch's Dataset and Dataloader classes.
    # define the NN models.
    
    # Define the Path to EEG data as a concatenation of:
    # 1) the root path
    # 2) the preprocessing pipeline
    if dataPath[-1] != os.sep:
        dataPath += os.sep
    if pipelineToEval[-1] != os.sep:
        eegpath = dataPath + pipelineToEval + os.sep
    else:
        eegpath = dataPath + pipelineToEval
    
    # Define the number of Channels to use. 
    # Basically 61 due to BIDSAlign channel system alignment.
    # Note that BIDSAlign DOES NOT delete any original channel by default.
    if rem_interp:
        if 'alzheimer' in taskToEval.casefold():
            Chan = 19
        elif taskToEval.casefold() == 'cognitive':
            Chan = 59
    else:
        Chan = 61
    
    # Define the sampling rate.
    freq = 125 if downsample else 250
    
    # Define the number of classes to predict.
    # All tasks are binary except the Alzheimer's one, 
    # which is a multi-class classification (Alzheimer vs FrontoTemporal vs Control)
    if taskToEval.casefold() == 'alzheimer':
        nb_classes = 3
    elif taskToEval.casefold() == 'cognitive':
        nb_classes = 4
    else:
        nb_classes = 2
    
    # For selfEEG's models instantiation
    Samples = int(freq*window)
    
    # Set the Dataset ID for glob.glob operation in SelfEEG's GetEEGPartitionNumber().
    # It is a single number for every task except for PD that merges two datasets
    if 'alzheimer' in taskToEval.casefold():
        datasetID = '10'
    elif taskToEval.casefold() == 'cognitive':
        datasetID = '19'
    
    # Set the class label in case of plot of functions
    if taskToEval.casefold() == 'alzheimer':
        class_labels = ['CTL', 'FTD', 'AD']
    elif taskToEval.casefold() == 'alzheimerca':
        class_labels = ['CTL', 'AD']
    elif taskToEval.casefold() == 'alzheimercf':
        class_labels = ['CTL', 'FTD']
    elif taskToEval.casefold() == 'alzheimeraf':
        class_labels = ['FTD', 'AD']
    elif taskToEval.casefold() == 'cognitive':
        class_labels = ['CTL', 'PD', 'PDD', 'PDMCI']
    
    # =====================================================
    #  Section 5: Define pytorch's Datasets and dataloaders
    # =====================================================
    
    # Now that everything is ready, let's define the pytorch's Datasets and dataloaders. 
    # The dataset is defined by using the selfEEG EEGDataset custom class, 
    # which includes an option to preload the entire dataset.
    
    # GetEEGPartitionNumber doesn't need the labels
    loadEEG_args = {
        'return_label': False, 
        'downsample': downsample, 
        'use_only_original': rem_interp,
        'apply_zscore': z_score
    }
    
    if taskToEval.casefold() == 'parkinson':
        glob_input = [datasetID_1 + '_*.pickle', datasetID_2 + '_*.pickle' ]
    else:
        glob_input = [datasetID + '_*.pickle']
    
    # calculate dataset length.
    # Basically it automatically retrieves all the partitions 
    # that can be extracted from each EEG signal
    EEGlen = dl.get_eeg_partition_number(
        eegpath,
        freq,
        window,
        overlap, 
        file_format = glob_input,
        load_function = loadEEG,
        optional_load_fun_args = loadEEG_args,
        includePartial = False if overlap == 0 else True,
        verbose = verbose
    )
    
    # Now we also need to load the labels
    loadEEG_args['return_label'] = True
    
    # Set functions to retrieve dataset, subject, and session from each filename.
    # They will be used by GetEEGSplitTable to perform a subject based split
    dataset_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[0])
    subject_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[1]) 
    session_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[2]) 
    
    # Now call the GetEEGSplitTable. Since Parkinson task merges two datasets
    # we need to differentiate between this and other tasks
    if taskToEval.casefold() == 'parkinson':
        # Remember
        # 1 --> 5 = EEG 3-Stim    //     2 --> 8 = UCSD
        train_id   = { 5: partition_list_1[foldToEval][0], 
                       8: partition_list_2[foldToEval][0]}
        val_id     = { 5: partition_list_1[foldToEval][1], 
                       8: partition_list_2[foldToEval][1]}
        test_id    = { 5: partition_list_1[foldToEval][2],
                       8: partition_list_2[foldToEval][2]}
        EEGsplit= dl.get_eeg_split_table(
            partition_table      = EEGlen,
            exclude_data_id      = None,  #[8], just checked if UCSD was useful 
            val_data_id          = val_id,
            test_data_id         = test_id, 
            split_tolerance      = 0.001,
            dataset_id_extractor = dataset_id_ex,
            subject_id_extractor = subject_id_ex,
            perseverance         = 10000
        )
        
    else:
        
        if taskToEval.casefold() == 'alzheimerca':
            exclude_id = f_id
        elif taskToEval.casefold() == 'alzheimercf':
            exclude_id = a_id
        elif taskToEval.casefold() == 'alzheimeraf':
            exclude_id = c_id
        else:
            exclude_id = None
        train_id   = partition_list[foldToEval][0]
        val_id     = partition_list[foldToEval][1]
        test_id    = partition_list[foldToEval][2]
        EEGsplit= dl.get_eeg_split_table(
            partition_table      = EEGlen,
            val_data_id          = val_id,
            test_data_id         = test_id,
            exclude_data_id      = exclude_id,
            split_tolerance      = 0.001,
            dataset_id_extractor = subject_id_ex,
            subject_id_extractor = session_id_ex,
            perseverance         = 10000
        )
    
    if verbose:
        print(' ')
        print('Subjects used for test')
        print(test_id)
        
    # Define Datasets and preload all data
    trainset = dl.EEGDataset(
        EEGlen, EEGsplit, [freq, window, overlap], 'train', 
        supervised             = True, 
        label_on_load          = True,
        load_function          = loadEEG,
        optional_load_fun_args = loadEEG_args
    )
    trainset.preload_dataset()
    
    valset = dl.EEGDataset(
        EEGlen, EEGsplit, [freq, window, overlap], 'validation',
        supervised             = True, 
        label_on_load          = True,
        load_function          = loadEEG,
        optional_load_fun_args = loadEEG_args
    )
    valset.preload_dataset()
    
    testset = dl.EEGDataset(
        EEGlen, EEGsplit, [freq, window, overlap], 'test',
        supervised             = True,
        label_on_load          = True,
        load_function          = loadEEG,
        optional_load_fun_args = loadEEG_args
    )
    testset.preload_dataset()
    
    # Convert to long if task is multiclass classification.
    # This avoids Value Errors during cross entropy loss calculation
    if ('alzheimer' in taskToEval.casefold()) or ('cognitive' in taskToEval.casefold()):
        if taskToEval.casefold() == 'alzheimerca':
            trainset.y_preload[trainset.y_preload==2] = 1
            valset.y_preload[valset.y_preload==2] = 1 
            testset.y_preload[testset.y_preload==2] = 1
        elif taskToEval.casefold() == 'alzheimeraf':
            trainset.y_preload -= 1
            valset.y_preload -= 1 
            testset.y_preload -= 1
        elif taskToEval.casefold() == 'alzheimeraf':
            pass
        else:
            trainset.y_preload = trainset.y_preload.to(dtype = torch.long)
            valset.y_preload   = valset.y_preload.to(dtype = torch.long)
            testset.y_preload  = testset.y_preload.to(dtype = torch.long)
          
    trainset.x_preload = trainset.x_preload.to(device=device)
    trainset.y_preload = trainset.y_preload.to(device=device)
    valset.x_preload = valset.x_preload.to(device=device)
    valset.y_preload = valset.y_preload.to(device=device)
    testset.x_preload = testset.x_preload.to(device=device)
    testset.y_preload = testset.y_preload.to(device=device)
    
    # Finally, Define Dataloaders
    # (no need to use more workers in validation and test dataloaders)
    trainloader = DataLoader(dataset = trainset, batch_size = batchsize,
                             shuffle = True, num_workers = workers)
    valloader = DataLoader(dataset = valset, batch_size = batchsize,
                           shuffle = False, num_workers = 0)
    testloader = DataLoader(dataset = testset, batch_size = batchsize,
                            shuffle = False, num_workers = 0)
    
    if verbose:
        # plot split statistics
        labels = np.zeros(len(EEGlen))
        for i in range(len(EEGlen)):
            path = EEGlen.iloc[i,0]
            with open(path, 'rb') as eegfile:
                EEG = pickle.load(eegfile)
            labels[i] = EEG['label']
        dl.check_split(EEGlen, EEGsplit, labels)
    
    # ===================================================
    #  Section 6: define the loss, model, and optimizer
    # ==================================================
    
    lossVal = None
    validation_loss_args = []
    if taskToEval.casefold()=="alzheimer" or ('cognitive' in taskToEval.casefold()):
        if subjectTrain:
            if appleLoss:
                lossFnc = subject_invariant_cross_entropy_apple
                lossVal = subject_invariant_cross_entropy_apple
            else:
                lossFnc = subject_invariant_cross_entropy
                lossVal = subject_invariant_cross_entropy
            validation_loss_args = {'ignore_subject': True}
        else:
            lossFnc = lossMulti
    else:
        if subjectTrain:
            if appleLoss:
                lossFnc = subject_invariant_binary_cross_entropy
                lossVal = subject_invariant_binary_cross_entropy
            else:
                lossFnc = subject_invariant_binary_cross_entropy_apple
                lossVal = subject_invariant_binary_cross_entropy_apple
            validation_loss_args = {'ignore_subject': True}
        else:
            lossFnc = lossBinary
            
    # SET SEEDS FOR REPRODUCIBILITY
    random.seed( seed )
    np.random.seed( seed )
    torch.manual_seed( seed )

    # define model
    if modelToEval.casefold() == 'eegnet':
        Mdl = zoo.EEGNet(nb_classes, Chan, Samples,
                         depthwise_max_norm=None, norm_rate=None, seed=seed)

    elif modelToEval.casefold() == 'shallownet':
        Mdl = zoo.ShallowNet(nb_classes, Chan, Samples, seed=seed)

    elif modelToEval.casefold() == 'deepconvnet':
        Mdl = zoo.DeepConvNet(nb_classes, Chan, Samples,
                              kernLength = 10, F = 25, Pool = 3,
                              stride = 3, batch_momentum = 0.1,
                              dropRate = 0.5, max_norm = None,
                              max_dense_norm = None, seed=seed)

    elif modelToEval.casefold() == 'atcnet':
        Mdl = zoo.ATCNet(nb_classes, Chan, Samples, freq,
                     max_norm = 2.0, tcn_max_norm=2.0,
                     batchMomentum=0.1, tcn_batchMom=0.1, seed=seed)

    elif modelToEval.casefold() == 'fbcnet':
        Mdl = zoo.FBCNet(nb_classes, Chan, Samples, freq, 
                      depthwise_max_norm=None, linear_max_norm=None, seed=seed)

    elif modelToEval.casefold() == 'xeegnet':
        Mdl = zoo.xEEGNet(nb_classes, Chan, Samples, freq, seed=seed)

    elif modelToEval.casefold() == 'eegconf':
        Mdl = zoo.EEGConformer(nb_classes, Chan, Samples, seed=seed)

    MdlBase = copy.deepcopy(Mdl)
    Mdl.to(device = device)
    Mdl.train()
    if verbose:
        print(' ')
        ParamTab = selfeeg.utils.count_parameters(Mdl, False, True, True)
        print(' ')
    
    if lr == 0:
        lr = GetLearningRate(modelToEval, taskToEval)
        if verbose:
            print(' ')
            print('used learning rate', lr)
    gamma = 0.995
    optimizer = torch.optim.Adam(Mdl.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)
    
    # Define selfEEG's EarlyStopper with large patience to act as a model checkpoint
    earlystop = selfeeg.ssl.EarlyStopping(
        patience = 15, 
        min_delta = 1e-04, 
        record_best_weights = True
    )
    
    # =============================
    #  Section 7: train the model
    # =============================
    loss_summary=train_model(
        model                 = Mdl,
        train_dataloader      = trainloader,
        epochs                = 1000,
        optimizer             = optimizer,
        loss_func             = lossFnc, 
        lr_scheduler          = scheduler,
        EarlyStopper          = earlystop,
        validation_dataloader = valloader,
        validation_loss_func  = lossVal,
        validation_loss_args  = validation_loss_args,
        verbose               = False,#verbose,
        device                = device,
        return_loss_info      = True
    )
 
    # ===============================
    #  Section 8: evaluate the model
    # ===============================
    earlystop.restore_best_weights(Mdl)
    Mdl.to(device=device)
    Mdl.eval()
    scores = get_performances(loader2eval    = testloader, 
                              Model          = Mdl, 
                              device         = device,
                              nb_classes     = nb_classes,
                              return_scores  = True,
                              verbose        = verbose,
                              plot_confusion = False,
                              class_labels   = class_labels
                             )
    
    training_loss_curve = []
    validation_loss_curve = []
    # Iterate through the dictionary, filtering out entries with None values
    for key, (train_loss, val_loss) in loss_summary.items():
        if train_loss is not None and val_loss is not None:
            training_loss_curve.append(train_loss)
            validation_loss_curve.append(val_loss)
    
    # Store the results in a new dictionary called 'scores'
    scores['training_loss_curve']   = training_loss_curve
    scores['validation_loss_curve'] = validation_loss_curve
    
    # ==================================
    #  Section 9: Save model and metrics
    # ==================================
    
    # Set the output path
    if taskToEval.casefold() == 'alzheimer':
        start_piece_mdl = 'AlzAllClassification/Models/'
        start_piece_res = 'AlzAllClassification/Results/'
        task_piece = 'alz'
    elif taskToEval.casefold() == 'alzheimerca':
        start_piece_mdl = 'AlzCAClassification/Models/'
        start_piece_res = 'AlzCAClassification/Results/'
        task_piece = 'alzca'
    elif taskToEval.casefold() == 'cognitive':
        start_piece_mdl = 'CognitiveClassification/Models/'
        start_piece_res = 'CognitiveClassification/Results/'
        task_piece = 'cgn'
    
    mdl_piece = modelToEval.casefold()

    if pipelineToEval.casefold() == 'raw':
        pipe_piece = 'raw'
    elif pipelineToEval.casefold() == 'filt':
        pipe_piece = 'flt'
    elif pipelineToEval.casefold() == 'ica':
        pipe_piece = 'ica'
    elif pipelineToEval.casefold() == 'icasr':
        pipe_piece = 'isr'
    
    if downsample:
        freq_piece = '125'
    else:
        freq_piece = '250'

    out_piece = str(outerFold+1).zfill(3)
    in_piece = str(innerFold+1).zfill(3)
    lr_piece = str(int(lr*1e6)).zfill(6)
    chan_piece = str(Chan).zfill(3)
    win_piece = str(round(window)).zfill(3)
    
    file_name = '_'.join(
        [task_piece, pipe_piece, freq_piece, mdl_piece, 
         out_piece, in_piece, lr_piece, chan_piece, win_piece
        ]
    )
    model_path = start_piece_mdl + file_name + '.pt'
    results_path = start_piece_res + file_name + '.pickle'
    
    if verbose:
        print('saving model and results in the following paths')
        print(model_path)
        print(results_path)
    
    # Save the model
    Mdl.eval()
    Mdl.to(device='cpu')
    torch.save(Mdl.state_dict(), model_path)
    
    # Save the scores
    with open(results_path, 'wb') as handle:
        pickle.dump(scores, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    if verbose:
        print('run complete')

import argparse
import glob
import itertools
import os
import pickle
import pandas as pd

def makeGrid(pars_dict):
        keys = pars_dict.keys()
        combinations = itertools.product(*pars_dict.values())
        ds = [dict(zip(keys, cc)) for cc in combinations]
        return ds

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.casefold() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.casefold() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0)" % (x,))
    return x

def positive_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0:
        raise argparse.ArgumentTypeError("%r not a positive value" % (x,))
    return x

def positive_int_nozero(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an integer" % (x,))

    if x < 0:
        raise argparse.ArgumentTypeError("%r not a positive value" % (x,))
    return x

def positive_int(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an integer" % (x,))

    if x < 0:
        raise argparse.ArgumentTypeError("%r not a positive value" % (x,))
    return x

def column_switch(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df.reindex(columns = i)
    return df


def gather_results(save = False, filename = None):
    
    metrics_list = [ 
        'accuracy_unbalanced', 'accuracy_weighted',
        'precision_micro',     'precision_macro',   'precision_weighted',
        'recall_micro',        'recall_macro',      'recall_weighted',
        'f1score_micro',       'f1score_macro',     'f1score_weighted',
        'rocauc_micro',        'rocauc_macro',      'rocauc_weighted',
        'cohen_kappa'
    ]
    piece_list = [
        'task', 'pipeline', 'sampling_rate', 'model', 
        'outer_fold', 'inner_fold', 'learning_rate',
        'channels', 'window',
        'subject_train', 'loss_type', 'subject_head'
    ]
    
    set_full = set(glob.glob('**/Results/*.pickle'))
    set_torm = set(glob.glob('Supplementary/Results/*.pickle'))
    file_list = list(set_full - set_torm)
    results_list = [None]*len(file_list)
    for i, path in enumerate(file_list):

        # Get File name
        file_name = path.split(os.sep)[-1]
        file_name = file_name[:-7]

        # Get all name pieces
        pieces = file_name.split('_')

        # convert to numerical some values
        for k in [2,4,5,6,7,8]:
            pieces[k] = int(pieces[k])
            if k == 6:
                pieces[k] = pieces[k]/1e6

        # open results
        with open(path, "rb") as f:
            mdl_res = pickle.load(f)

        # append results
        for metric in metrics_list:
            pieces.append(mdl_res[metric])

        # final list
        results_list[i] = pieces

    # convert to DataFrame and swap two columns for convenience
    results_table = pd.DataFrame(results_list, columns= piece_list + metrics_list)
    results_table = column_switch( results_table, 'model', 'sampling_rate')
    results_table.sort_values(
        ['model', 'task','pipeline','inner_fold','outer_fold'],
        ascending=[True, True, True, True, True],
        inplace=True
    )

    # store if required
    if save:
        if filename is not None:
            if filename[:-3] == 'csv':
                results_table.to_csv(filename, index=False)
            else:
                results_table.to_csv(filename + '.csv', index=False)
        results_table.to_csv('ResultsTable.csv', index=False)
    return results_table


def GetLrDict():
    lr_dict = {
        'eegnet': {
            'eyes': 5e-04,
            'parkinson': 1e-04,
            'alzheimer': 7.5e-04,
            'motorimagery': 1e-03,
            'sleep': 1e-03
        },
        'shallownet': {
            'eyes': 1e-03,
            'parkinson': 2.5e-04, #2.5e-05
            'alzheimer': 5e-05,
            'motorimagery': 7.5e-04,
            'sleep': 5e-05
        },
        'deepconvnet': {
            'eyes': 7.5e-04,
            'parkinson': 2.5e-04,
            'alzheimer': 7.5e-04,
            'motorimagery': 7.5e-04,
            'sleep': 2.5e-04
        },
        'fbcnet': {
            'eyes': 7.5e-04,
            'parkinson': 2.5e-04,
            'alzheimer': 7.5e-05,
            'motorimagery': 1e-3,
            'sleep': 1e-04
        }
    }
    lr_dict['shallownet_custom'] = lr_dict['shallownet']
    lr_dict['psdnet'] = lr_dict['eegnet']
    lr_dict['psdnet4'] = lr_dict['eegnet']
    lr_dict['psdnet2'] = lr_dict['shallownet']
    lr_dict['psdnet3'] = lr_dict['shallownet']
    lr_dict['psdnet5'] = lr_dict['shallownet']
    lr_dict['psdnet6'] = lr_dict['shallownet']
    lr_dict['psdnet7'] = lr_dict['shallownet']
    lr_dict['psdnet8'] = lr_dict['shallownet']
    lr_dict['psdnet9'] = lr_dict['shallownet']
    lr_dict['shallownet2'] = lr_dict['shallownet']
    lr_dict['shallownet3'] = lr_dict['shallownet']
    lr_dict['shallownet4'] = lr_dict['shallownet']
    lr_dict['shallownet5'] = lr_dict['shallownet']
    lr_dict['shallownet6'] = lr_dict['shallownet']
    lr_dict['shallownet7'] = lr_dict['shallownet']
    lr_dict['shallownet8'] = lr_dict['shallownet']
    lr_dict['shallownet9'] = lr_dict['shallownet']
    return lr_dict

def GetLearningRateString(model, task):
    model_conversion_dict = {
        'egn': 'eegnet', 'shn': 'shallownet',
        'sh2': 'shallownet2', 'sh3': 'shallownet3',
        'sh4': 'shallownet4', 'sh5': 'shallownet5',
        'sh6': 'shallownet6', 'sh7': 'shallownet7',
        'sh8': 'shallownet8', 'sh9': 'shallownet9',
        'dcn': 'deepconvnet', 'fbc': 'fbcnet',
        'psd': 'psdnet', 'ps2': 'psdnet2',
        'ps3': 'psdnet3', 'ps4': 'psdnet4',
        'ps5': 'psdnet5', 'ps6': 'psdnet2',
        'ps7': 'psdnet3', 'ps8': 'psdnet4',
        'ps9': 'psdnet5'
    }
    task_conversion_dict = {
        'eye': 'eyes', 'alz': 'alzheimer',
        'al1': 'alzheimer', 'al2': 'alzheimer', 'al3': 'alzheimer',  
        'mmi': 'motorimagery', 'pds': 'parkinson',
        'slp': 'sleep'
    }
    if len(model)==3:
        model = model_conversion_dict.get(model)
    if len(task)==3:
        task = task_conversion_dict.get(task)
    lr_dict = GetLrDict()
    if "alzheimer" in task.casefold():
        task = "alzheimer"
    elif "cognitive" in task.casefold():
        task = "cognitive"
    lr = lr_dict.get(model).get(task)
    lr = str(int(lr*1e6)).zfill(6)
    return lr
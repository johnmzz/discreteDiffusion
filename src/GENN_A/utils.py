import math
import numpy as np
import torch
from texttable import Texttable

from os.path import basename, isfile
from os import makedirs
from glob import glob
import networkx as nx
import json


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    rows = [["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    t.add_rows(rows)
    print(t.draw())

def sorted_nicely(l):
    """
    Sort file names in a fancy way.
    The numbers in file names are extracted and converted from str into int first,
    so file names can be sorted based on int comparison.
    :param l: A list of file names:str.
    :return: A nicely sorted file name list.
    """

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)

def get_file_paths(dir, file_format='json'):
    """
    Return all file paths with file_format under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return paths: The paths of all required files.
    """
    dir = dir.rstrip('/')
    paths = sorted_nicely(glob(dir + '/*.' + file_format))
    return paths

def iterate_get_graphs(dir, file_format):
    """
    Read networkx (dict) graphs from all .gexf (.json) files under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return graphs: Networkx (dict) graphs.
    """
    assert file_format in ['gexf', 'json', 'onehot', 'anchor']
    graphs = []
    for file in get_file_paths(dir, file_format):
        gid = int(basename(file).split('.')[0])
        if file_format == 'gexf':
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            if not nx.is_connected(g):
                raise RuntimeError('{} not connected'.format(gid))
        elif file_format == 'json':
            # g is a dict
            g = json.load(open(file, 'r'))
            g['gid'] = gid
        elif file_format in ['onehot', 'anchor']:
            # g is a list of onehot labels
            g = json.load(open(file, 'r'))
        graphs.append(g)
    return graphs

def load_all_graphs(data_location, dataset_name):
    graphs = iterate_get_graphs(data_location + "json_data/" + dataset_name + "/train", "json")
    train_num = len(graphs)
    graphs += iterate_get_graphs(data_location + "json_data/" + dataset_name + "/test", "json")
    test_num = len(graphs) - train_num
    val_num = test_num
    train_num -= val_num
    return train_num, val_num, test_num, graphs

def load_labels(data_location, dataset_name):
    path = data_location + "json_data/" + dataset_name + "/labels.json"
    global_labels = json.load(open(path, 'r'))
    features = iterate_get_graphs(data_location + "json_data/" + dataset_name + "/train", "onehot") \
             + iterate_get_graphs(data_location + "json_data/" + dataset_name + "/test", "onehot")
    print('Load one-hot label features (dim = {}) of {}.'.format(len(global_labels), dataset_name))
    return global_labels, features

def load_ged(ged_dict, data_location='', dataset_name='AIDS', file_name='TaGED.json'):
    '''
    list(tuple)
    ged = [(id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, [best_node_mapping])]

    id_1 and id_2 are the IDs of a graph pair, e.g., the ID of 4.json is 4.
    The given graph pairs satisfy that n1 <= n2.

    ged_value = ged_nc + ged_in + ged_ie
    (ged_nc, ged_in, ged_ie) is the type-aware ged following the setting of TaGSim.
    ged_nc: the number of node relabeling
    ged_in: the number of node insertions/deletions
    ged_ie: the number of edge insertions/deletions

    [best_node_mapping] contains 10 best matching at most.
    best_node_mapping is a list of length n1: u in g1 -> best_node_mapping[u] in g2

    return dict()
    ged_dict[(id_1, id_2)] = ((ged_value, ged_nc, ged_in, ged_ie), best_node_mapping_list)
    '''
    path = "{}json_data/{}/{}".format(data_location, dataset_name, file_name)
    TaGED = json.load(open(path, 'r'))
    for (id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, mappings) in TaGED:
        ta_ged = (ged_value, ged_nc, ged_in, ged_ie)
        ged_dict[(id_1, id_2)] = (ta_ged, mappings)

def load_features(data_location, dataset_name, feature_name):
    features = iterate_get_graphs(data_location + "json_data/" + dataset_name + "/train", feature_name) \
             + iterate_get_graphs(data_location + "json_data/" + dataset_name + "/test", feature_name)
    feature_dim = len(features[0][0])
    print('Load {} features (dim = {}) of {}.'.format(feature_name, feature_dim, dataset_name))
    return feature_dim, features



def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    temp = prediction.argsort()
    r_prediction = np.empty_like(temp)
    r_prediction[temp] = np.arange(len(prediction))

    temp = target.argsort()
    r_target = np.empty_like(temp)
    r_target[temp] = np.arange(len(target))
    
    return rank_corr_function(r_prediction, r_target).correlation

def calculate_prec_at_k(k, prediction, groundtruth):
    """
    Calculating precision at k.
    """
    best_k_pred = prediction.argsort()[-k:]
    best_k_gt = groundtruth.argsort()[-k:]
    
    return len(set(best_k_pred).intersection(set(best_k_gt))) / k

def denormalize_sim_score(g1, g2, sim_score):
    """
    Converts normalized similar into ged.
    """
    return denormalize_ged(g1, g2, -math.log(sim_score, math.e))

def denormalize_ged(g1, g2, nged):
    """
    Converts normalized ged into ged.
    """
    return round(nged * (g1.num_nodes + g2.num_nodes) / 2)
        
def to_directed(edge_index):
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)

def aids_labels(g):
    types = [
        'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'F',
        'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
        'Sb', 'Se', 'Ni', 'Te'
    ]
    
    return [types[i] for i in g.x.argmax(dim=1).tolist()]

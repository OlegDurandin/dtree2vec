import pickle
import networkx as nx

def load_pickled_file(one_file):
    with open(one_file, 'rb') as f:
        obj = pickle.load(f)
    return obj

def convert_conll_tree_to_nx(list_of_conll_tokens):
    list_of_edges = []
    dict_of_nodes = {}

    for one_token in list_of_conll_tokens:
        if type(one_token['id']) is tuple:
            continue
        current_id = one_token['id']
        dict_of_nodes[current_id] = {
            'word' : one_token['form'].lower(),
            'pos' : one_token['upostag']
        }
        current_edge = ((one_token['head'], current_id, {'dep' : one_token['deprel']}))
        list_of_edges.append(current_edge)
    DG = nx.DiGraph()
    DG.add_edges_from(list_of_edges)
    for index in dict_of_nodes:
        DG.add_node(index, **dict_of_nodes[index])
    DG.add_node(0, pos = 'ROOT', word = 'ROOT')
    return DG

def conversion_dependency_tree_to_graph(dependency_tree):
    # Function for conversion from dependency tree (SpaCy)
    # to DiGraph NetworkX structure
    list_of_edges = []
    dict_of_nodes = {}
    for i, word in enumerate(dependency_tree):
        if word == word.head:
            head_index = 0
        else:
            head_index = word.head.i + 1
        #current_edge = ((i + 1, head_index, {'dep': word.dep_}))
        current_edge = ((head_index, i + 1,  {'dep': word.dep_}))
        list_of_edges.append(current_edge)
        dict_of_nodes[i + 1] = {'word': word.text,
                                'pos': word.pos_,
                                'tag': word.tag_}

    DG = nx.DiGraph()
    DG.add_edges_from(list_of_edges)
    for index in dict_of_nodes:
        DG.add_node(index, **dict_of_nodes[index])
    DG.add_node(0, pos='ROOT')
    return DG

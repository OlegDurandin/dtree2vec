from typing import List
import networkx as nx
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# This file contain realization of different representation methods
# as a part of graph2vec for dependency tree adaptation


TreeVector = List[nx.DiGraph]
GraphRepresentationVector = List[TaggedDocument]


class SimplePathExtractor:
    def __init__(self):
        self.fitted = False
        self.node_attr = 'pos'

    def feature_extraction(self, graph_instance: nx.DiGraph):
        total_paths = []
        for first_node in graph_instance.nodes():
            for second_node in graph_instance.nodes():
                if first_node == second_node:
                    continue
                all_simple_paths = list(nx.all_simple_paths(graph_instance,
                                                            source=first_node,
                                                            target=second_node))
                if all_simple_paths:
                    for one_simple_path in all_simple_paths:
                        str_representation_of_path = ''
                        for index, one_step in enumerate(one_simple_path[:-1]):
                            str_representation_of_path += graph_instance.node[one_step].get(self.node_attr, '')
                            str_representation_of_path += '_' + graph_instance.get_edge_data(one_simple_path[index],
                                                                                             one_simple_path[
                                                                                                 index + 1])[
                                'dep'].lower()
                            str_representation_of_path += '_'
                        str_representation_of_path += graph_instance.node[one_simple_path[-1]].get(self.node_attr, '')
                        total_paths.append(str_representation_of_path)
        return total_paths


class ContractedFeatureExtractor:
    def __init__(self, dependency_activity: bool):
        self.dependency = dependency_activity

    def feature_extraction(self, graph_instance: nx.DiGraph):
        dx_second = graph_instance.copy()
        mffe = [one_data['pos'] for one_node, one_data in dx_second.nodes(data=True)]
        zero_out_degree_nodes = [node for node, out_degree in dx_second.out_degree() if out_degree == 0]

        iteration_number = 0
        while (zero_out_degree_nodes):
            for zero_node in zero_out_degree_nodes:
                in_edge = dx_second.in_edges(zero_node, data=True)
                if len(in_edge) > 1:
                    print("Critical Error")
                for edge in in_edge:
                    edge_from_node = edge[0]
                    edge_data = edge[2]['dep']
                dependent_node_pos = dx_second.nodes[zero_node]['pos']
                head_node_pos = dx_second.nodes[edge_from_node]['pos']
                dx_second = nx.contracted_nodes(dx_second, edge_from_node, zero_node, self_loops=False)
                if self.dependency:
                    dx_second.nodes[edge_from_node]['pos'] = '{}_{}_{}'.format(head_node_pos, edge_data,
                                                                               dependent_node_pos)
                else:
                    dx_second.nodes[edge_from_node]['pos'] = '{}_{}'.format(head_node_pos, dependent_node_pos)
            if len(dx_second.nodes()) > 1:
                zero_out_degree_nodes = [node for node, out_degree in dx_second.out_degree() if out_degree == 0]
            else:
                zero_out_degree_nodes = []
            iteration_number += 1
            mffe.extend([one_data['pos'] for one_node, one_data in dx_second.nodes(data=True)])
        return mffe


class FixedContractedFeatureExtractor:
    def __init__(self, dependency_activity: bool):
        self.dependency = dependency_activity

    def feature_extraction(self, graph_instance: nx.DiGraph):
        dx_second = graph_instance.copy()
        mffe = [one_data['pos'] for one_node, one_data in dx_second.nodes(data=True)]
        zero_out_degree_nodes = [node for node, out_degree in dx_second.out_degree() if out_degree == 0]
        iteration_number = 0
        while (zero_out_degree_nodes):
            dx_current_tree = dx_second.copy()

            for zero_node in zero_out_degree_nodes:
                in_edge = dx_current_tree.in_edges(zero_node, data=True)
                if len(in_edge) > 1:
                    print("Critical Error")
                for edge in in_edge:
                    edge_from_node = edge[0]
                    edge_data = edge[2]['dep']

                dependent_node_pos = dx_current_tree.nodes[zero_node]['pos']
                head_node_pos = dx_current_tree.nodes[edge_from_node]['pos']
                mffe.append('{}_{}_{}'.format(head_node_pos, edge_data, dependent_node_pos))

                dependent_node_pos_modification = dx_second.nodes[zero_node]['pos']
                head_node_pos_modification = dx_second.nodes[edge_from_node]['pos']

                dx_second = nx.contracted_nodes(dx_second, edge_from_node, zero_node, self_loops=False)
                dx_second.nodes[edge_from_node]['pos'] = '{}_{}_{}'.format(head_node_pos_modification, edge_data,
                                                                           dependent_node_pos_modification)

            if len(dx_second.nodes()) > 1:
                zero_out_degree_nodes = [node for node, out_degree in dx_second.out_degree() if out_degree == 0]
            else:
                zero_out_degree_nodes = []
            iteration_number += 1
            mffe.extend([one_data['pos'] for one_node, one_data in dx_second.nodes(data=True)])
        return mffe


class WLExtractor:
    def __init__(self,
                 round: int,
                 dependency_activity: bool):
        self.round = round
        self.dependency_activity = dependency_activity

    def graph_reader(self, graph_instance: nx.DiGraph):
        features = {node_number: node_data['pos'] for node_number, node_data in graph_instance.nodes(data=True)}
        return graph_instance, features

    def feature_extraction(self, graph_instance: nx.DiGraph):
        self.current_graph = graph_instance
        self.features = {node_number: node_data['pos'] for node_number, node_data in
                         self.current_graph.nodes(data=True)}
        self.extracted_features = [str(v) for k, v in self.features.items()]
        self.do_recursions()
        return self.extracted_features

    def do_recursions(self):
        for iteration_index in range(self.round):
            self.features = self.do_a_recursion()

    def do_a_recursion(self):
        new_features = {}
        for node in self.current_graph.nodes:
            nebs = list(self.current_graph.neighbors(node))
            if self.dependency_activity:
                potential_features = '_'.join(
                    [self.features[node]] + [(self.current_graph.get_edge_data(node, neib_num)['dep'].lower() + '_' +
                                              self.features[neib_num]) for neib_num in nebs])
            else:
                potential_features = '_'.join(
                    [self.features[node]] + [self.features[neib_num] for neib_num in nebs])
            new_features[node] = potential_features
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features


class DepTreeEmbedding:
    def __init__(self,
                 extractorInstance,
                 doc2vecargs: dict
                 ):
        self.Extractor = extractorInstance
        self.doc2vecargs = doc2vecargs
        self.workers = self.doc2vecargs['workers']
        self.model_fit = False
        self.document_collections = []

    def feature_extractor_with_index(self, graph_instance, index):
        return TaggedDocument(words=self.Extractor.feature_extraction(graph_instance), tags=["g_" + str(index)])

    def prepare_batch_of_docs(self, treeList: TreeVector) -> GraphRepresentationVector:
        # treeList - list of dependency graphs (as NetworkX Digraph objects)
        document_collections = Parallel(n_jobs=self.workers)(
            delayed(self.feature_extractor_with_index)(g, i) for i, g in tqdm(enumerate(treeList)))
        #        document_collections = [self.feature_extractor_with_index(g, i) for i, g in tqdm(enumerate(treeList))]
        return document_collections

    def light_weight_feature_extractor(self, treeList: TreeVector):
        document_collections = Parallel(n_jobs=self.workers)(
            delayed(self.Extractor.feature_extraction)(g) for g in tqdm(treeList))
        return document_collections

    def fit(self, treeList: TreeVector):
        self.document_collections = self.prepare_batch_of_docs(treeList)
        self.model = Doc2Vec(self.document_collections,
                             **self.doc2vecargs)
        self.model_fit = True
        self.count_of_trains = len(self.document_collections)

    def predict(self, treeList: TreeVector):
        self.document_collections = self.light_weight_feature_extractor(treeList)
        list_of_vectors = [self.model.infer_vector(one_doc) for one_doc in self.document_collections]
        return list_of_vectors

    def save_str_representation(self, path):
        outFile = open(path, 'w')
        for one_doc in self.document_collections:
            strRepresentation = ' '.join(one_doc.words)
            outFile.write(strRepresentation + '\n')
        outFile.close()

    def save_embeddings(self, path):
        outList = []
        for index in range(self.count_of_trains):
            outList.append([int(index)] + list(self.model.docvecs['g_{}'.format(index)]))

        outDF = pd.DataFrame(outList,
                             columns=['Type'] + ["x_" + str(dim_index) for dim_index in range(self.model.vector_size)])
        outDF = outDF.sort_values(['Type'])
        outDF.to_csv(path, index=None)

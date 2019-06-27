from tqdm import tqdm

import glob
import os
import pickle
import re

from dep_tree_embedding import WLExtractor, SimplePathExtractor, ContractedFeatureExtractor
from dep_tree_embedding import DepTreeEmbedding


from settings import doc2vec_arguments
from settings import CONLL_MODELS_FOLDER_NAME, MODEL_DIRECTORY

# Reading files, that created in conll_processor.py
# Processing them with different methods of representation dependency trees

representation_types = [
    ('wl_kernel_2_with_dependency', WLExtractor(2, True)),
    ('wl_kernel_2_without_dependency', WLExtractor(2, False)),
    ('path_extractor', SimplePathExtractor()),
    ('contracted_with_dependency', ContractedFeatureExtractor(True)),
    ('contracted_without_dependency', ContractedFeatureExtractor(False))

]

# If folder with results of embeddings doesn't exists, create it
if not os.path.exists(MODEL_DIRECTORY):
    os.makedirs(MODEL_DIRECTORY)

all_model_files = glob.glob('.//{}//*'.format(CONLL_MODELS_FOLDER_NAME))
for current_model_file in all_model_files:
    m = re.search('conll_parsed_(.+?)_language', current_model_file)
    if m:
        current_language = m.group(1)
        current_base_name = os.path.basename(current_model_file)
    else:
        # All that not consist with pattern - excluded
        continue

    print('Loading pickle file: {}'.format(current_model_file))
    list_of_nx_graphs = pickle.load(open(current_model_file, 'rb'))
    print('Pickle loaded, we have {} sentences'.format(len(list_of_nx_graphs)))


    for current_representation_type in representation_types:
        print('Current representation learning: {}'.format(current_representation_type[0]))
        extractorInstance = current_representation_type[1]
        embeddingInstance = DepTreeEmbedding(extractorInstance, doc2vec_arguments)
        embeddingInstance.fit(map(lambda x : x['tree'], list_of_nx_graphs))

        path_to_csv = os.path.join(MODEL_DIRECTORY, '{}_vectors_dim_{}_{}.csv'.format(
                                              current_representation_type[0],
                                              doc2vec_arguments['size'],
                                              current_language))
        embeddingInstance.save_embeddings(path_to_csv)
        embeddingInstance.save_str_representation(os.path.join(MODEL_DIRECTORY, '{}_str_representation_{}.txt'.format(
                                              current_representation_type[0],
                                              current_language)))

        pickle.dump(
            tuple([extractorInstance, embeddingInstance]),
            open(os.path.join(MODEL_DIRECTORY, 'FULL_{}_{}.model'.format(current_representation_type[0], current_language)), 'wb'))

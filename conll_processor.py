# Main file - doing preprocessing of CoNLL files
# First, check up count of languages
# Second, for each language read conll file, parsed it
# Thirdly, convert to NetworkX format

# Author: Oleg Durandin


import glob
import os
import pickle
from conllu import parse
import networkx as nx

from settings import CONLL_MODELS_FOLDER_NAME
from utils import convert_conll_tree_to_nx





all_conll_files = glob.glob('.//conll_files//*.conllu')
if not os.path.exists(CONLL_MODELS_FOLDER_NAME):
    os.makedirs(CONLL_MODELS_FOLDER_NAME)

languages = set([os.path.basename(base_conllu_name).split('_')[0] for base_conllu_name in all_conll_files])

print('We have: {} languages'.format(len(languages)))
for current_language in languages:
    print('Processing {} language'.format((current_language)))
    list_of_sentences_for_current_language = []
    for current_conllu_file in glob.glob('.//conll_files//{}*.conllu'.format(current_language)):
        print('Processing : {}'.format(os.path.basename(current_conllu_file)))
        conll_file = open(current_conllu_file, 'r', encoding='utf-8')
        text_of_file = conll_file.read()
        list_of_sentences_in_conll = parse(text_of_file)
        for one_sentence in list_of_sentences_in_conll:
            list_of_sentences_for_current_language.append(
                {'sentence' : one_sentence.metadata['text'],
                 'tree' : convert_conll_tree_to_nx(one_sentence)
                 }
            )

    pickle.dump(list_of_sentences_for_current_language,
                open(os.path.join(CONLL_MODELS_FOLDER_NAME, 'conll_parsed_{}_language'.format(current_language)), 'wb'))



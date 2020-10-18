# dtree2vec
This project contain implementation the system, proposed in the "Adapting the Graph2Vec Approach to Dependency Trees for NLP Tasks" (O. Durandin, A. Malafeev), and presented in the AIST-2019 Conference


### Requirements

Install next libraries:
* pip install conllu 
* pip install networkx
* pip install gensim

### How to use

#### conll_processor

В директории **conll_files** размещены conll файлы, которые используются, чтобы создать набор деревьев синтаксического разбора. Запуск скрипта **conll_processor.py** приведет созданию папки CoNLL_Models, в которой будут содержаться файлы со списком деревьев.
Они представляют собой pickle файлы в бинарном формате, и будут использованы в других скриптах.

#### conll_deptreeembedding
Звпуск файла conll_deptreeembedding приведет к тому, что он возьмет из директории CoNLL_Models бинарные файлы со списком деретев и создаст в папке EXPERIMENTAL_EMBEDDING_DEBUG результирующие файлы.
Векторные репрезентации хранятся в файлых csv и txt, model файл необходим для создания новых файлов, то есть чтобы перевести новые синтаксические деревья в векторный формат.


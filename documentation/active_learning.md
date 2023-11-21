# Train a neural ranking/retrieval model with active learning

Our main script for training a neural ranking/retrieval model with an active learning strategy is [``al_pipe.py``](/matchmaker/al_pipe.py)

``al_pipe.py`` allows you to train a neural ranking/retrieval model with actively selecting the samples to be annotated. 
In each iteration the training set is incrementally increased and a neural ranking/retrieval model is trained on the increased training set.
The resulting model is used for actively selecting the next samples to be added to the training set.

There are 3 [selection strategies implemented](/matchmaker/active_learning/selection_strategies_al.py):
uncertainty-based selection, diversity-based selection, and QBC [[1]](#1).

|**AL strategy**       | **Config requirements** |
| ------------- |:-------------          |
|1) uncertainty      | *uncertainty_cutoff* decision boundary to determine which samples are uncertain, has to be between [0,1], 0.5 on default  |
|2) diversity        | *clustering_config* config for creating the clusters for diversity-based sampling |
|3) QBC              | *no_comittee* number of committee members, *m_percentage_subset* share of training set to train one member (e.g. 0.8 equals 80 of the training set) |

- Trains and evaluates a ranking/retrieval model on a subset of the training data in each iteration, for evaluating a retrieval model with dense retrieval we refer to [``dense_retrieval.py``](/matchmaker/dense_retrieval.py)
- Selects training samples to be added to the training set, calculates and stores the annotation effort for adding the training samples
- Allows to continue experiments from intermediate trained models
- Every start of ``al_pipe.py`` creates a new experiment folder (copies the current code & config) and saves the results 


## CLI of al_pipe.py

To use ``al_pipe.py`` you need to set the config paths for the [data config]((/config/train/data/example-minimal-al-dataset.yaml)) (contains the paths to the data files), 
the model config (which ranking/retrieval model is trained, options between [bert_cat.yaml](/config/train/models/bert_cat.yaml) (MonoBERT), [colbert.yaml](/config/train/models/colbert.yaml) and 
[bert_dot.yaml](/config/train/models/bert_dot.yaml) (DPR) and the [active learning config](/config/train/modes/active_learning.yaml), a run-name, and optional config-overwrites.

````
python matchmaker/al_pipe.py --config-file config/train/defaults.yaml config/train/data/example-minimal-al-dataset.yaml config/train/models/bert_cat.yaml config/train/modes/active_learning.yaml --run-name <the_experiment_name> --config-overwrites "<optional override1>,<optional override2> ..."
````

Minimal usage example with the configs from the upper CLI command.

## Config requirements

Set certain hyperparameters for active learning in the [active learning config](/config/train/modes/active_learning.yaml):

- validate_every_n_batches: How often to validate during the training process, needs to be smaller than total number of steps in training, if not then no checkpoint of final model is stored (only in evaluation a checkpoint and evaluation metrics of the model are stored)
- training_step_write: Every n step that the training loss is written to the loss output file
- train_subset: True if a subset of the training set is used, False if the whole training dataset is used, then train_data_size is ignored
- train_data_size: Size of the training data that the model is trained on
- al_strategy: Select the active learning strategy, select from ['diversity', 'qbc', 'uncertainty']
- no_iterations_al: Number of iterations in the active learning process
- no_queries_added_iteration: Number of samples added to the training set per iteration in active learning
- al_selection_annotation: Determines which positive document is selected in the training sample, select from ['first', 'random']
- continue_final_training: If True, then continue with one final training at the end of the iterations, else False

Specific hyperparameters for different active learning strategies:
- for Uncertainty-based selection:
    - decision boundary cutoff for uncertainty-based selection: uncertainty_cutoff: 0.5, decision boundary to determine which samples are uncertain, has to be between [0,1], 0.5 per default
- for Diversity-based selection:
    - clustering_config: 'config/dense_retrieval/example-al-diversity-clustering.yaml', config for creating the clusters for diversity-based sampling
- for Query-by-Committee (QBC) selection:
    - no_comittee: 2, Number of committee members
    - m_percentage_subset: 0.8, Share of training set to train one member on

Requirements for the [data config](/config/train/data/example-minimal-al-dataset.yaml):

- Set the paths to the training, query, collection and qrels file, all files need to be in tsv format:
    - train_tsv: "/../train.tsv", needs to contain triples with query_text \t pos_text \t neg_text \n
    - query_tsv: "/../queries.train.tsv", needs to contain queries with query_id \t query_text \n
    - collection_tsv: '/../collection.tsv', needs to contain the collection with doc_id \t doc_text \n
    - qrels_train: "/../qrels.train.tsv", needs to contain the qrels for the training queries with query_id \t 0 \t doc_id \t rel_grade(0/1) \n

- Set the path to the read in the BM25 scores for the active learning methods
    - test: 
        - train_top100_bm25: (IMPORTANT THAT THIS NAME STAYS THE SAME AS IT IS READ INTO IN al_pipe.py FOR READING THE SCORES)
            - tsv: "/../queries.train.bm25.reranking.tsv", needs to contain the BM25 top 100 for reranking with query_id \t doc_id \t query_text \t doc_text \n
            - qrels: "/../qrels.train.tsv", needs to contain the qrels for the training queries with query_id \t 0 \t doc_id \t rel_grade(0/1) \n
            - binarization_point: 1
            - save_secondary_output: True

Requirements for the clustering config for diversity-based selection
- cluster_queries_tsv: /data/queries.train.tsv, path to queries which are embedded and then clustered

## Continue training from intermediate checkpoints

In case the training is terminated/killed within the iterations, you can continue the training. There are 2 cases from which you can continue: 1) The last iteration is fully finished or 2) Within the last iteration the ranker is trained and evaluated, but the new training samples have not been selected with the AL strategy and have not been added to the training set.

1) Continue the training in case the previous iteration (in this example iteration 1) is finished and the new training samples are already added to the subset training file in the run folder

You can setup for continuing the training, by adding following lines to the config.yaml in the run folder of the experiment:

````
iteration_continue: 2 # iteration from where the training is resumed
continue_train_file: "/run-folder-experiment/triples.train.subset.5000.tsv" # path to the subset training file of the run folder
````

2) Continue the training in case the iteration is not finished: the training samples to be added to the subset training file have not been added, but the ranker 
has been trained and evaluated:

You can setup for continuing the training, by adding following lines to the config.yaml in the run folder of the experiment:

````
iteration_continue: 2 # iteration from where the training is resumed
continue_train_file: "/run-folder-experiment/triples.train.subset.5000.tsv" # path to the subset training file of the run folder
commitee_members: [" "] # for QBC this needs to be a list of the members which are trained already, for uncertainty/diversity-based sampling this needs to be a string "" with the path to the trained ranker
````

Finally continue the training with following CLI command by adding --continue-folder and the path to the run folder to the command:

````
python matchmaker/al_pipe.py --config-file config/train/defaults.yaml config/train/data/example-minimal-al-dataset.yaml config/train/models/bert_cat.yaml config/train/modes/active_learning.yaml --run-name <the_experiment_name> --config_overwrites "<optional override1>,<optional override2> ... --continue-folder <run-folder-experiment>"
````

## References
<a id="1">[1]</a> 
Cai, Peng et al. (2011). 
Relevant knowledge helps in choosing right teacher: active query selection for ranking adaptation. 
SIGIR '11: Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval.
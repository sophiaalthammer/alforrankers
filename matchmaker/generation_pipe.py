import argparse
import sys
import io
import selectors
#from matchmaker.utils.config import *
import yaml
from typing import Dict, List
import os
import subprocess


def get_config(config_path: List[str]) ->Dict[str, any]:
    cfg = {}
    for path in config_path:
        with open(os.path.join(os.getcwd(), path), 'r') as ymlfile:
            cfg.update(yaml.load(ymlfile, Loader=yaml.FullLoader))
    _auto_config_filler(cfg)
    return cfg

_auto_config_info = {
    ("model_input_type", "model"): [
        (["bert_cat","bert_cls"], "concatenated"),
        (["bert_tower","bert_dot","bert_dot_qa", "TK", "TKL", "ColBERT","ColBERTer","ColBERT_v2","PreTTR","IDCM"], "independent")
    ],
    ("token_embedder_type", "model"): [
        (["bert_cat","bert_cls"], "bert_cat"),
        (["bert_tower","bert_dot","bert_dot_qa", "ColBERT","ColBERTer","ColBERT_v2","PreTTR","IDCM"], "bert_dot"),
        (["TK", "TKL"], "embedding")
    ]
}

def _auto_config_filler(config: Dict[str, any]):
    for (auto_set, auto_switch), cases in _auto_config_info.items():
        if config.get(auto_set, "") == "auto":
            success = False
            switch_target = config.get(auto_switch, "")
            for case, value in cases:
                if switch_target in case:
                    config[auto_set] = value
                    success = True
                    break
            if not success:
                raise Exception("Could not fill in auto config for: " + str(auto_set)+" where switch is based on: " + str(switch_target))

def execute_and_return_run_folder(run_args):
    p = subprocess.Popen(run_args, universal_newlines=True,stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    output, err = p.communicate()
    run_folder = output.strip().split('\n')[-1]
    return run_folder


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--warmstart_model_path', action='store', dest='warmstart_model_path',
                        help='training output text file location', required=False)
    parser.add_argument('--run_name', action='store', dest='run_name',
                        help='training output text file location', required=True)
    parser.add_argument('--config-file', nargs='+', action='store', dest='config_file',
                        help='config file with all hyper-params & paths', required=True)
    parser.add_argument('--machine', action='store', dest='machine',
                        help='training output text file location', required=False, choices=['dl', 'tr', 'ol'])
    parser.add_argument('--query-path', action='store', dest='query_path',
                        help='Either true or false', required=False)
    parser.add_argument('--candidate-path', action='store', dest='candidate_path',
                        help='Either true or false', required=False)
    parser.add_argument('--scoring-model', action='store', dest='scoring_model',
                        help='Either true or false', required=False)
    parser.add_argument('--scores-file', action='store', dest='scores_file',
                        help='Either true or false', required=False)
    parser.add_argument('--triples-file', action='store', dest='triples_file',
                        help='Either true or false', required=False)
    parser.add_argument('--beam', action='store', dest='beam',
                        help='Either true or false', required=False)
    args = parser.parse_args()

    #args = get_parser().parse_args()


    config = get_config(args.config_file)


    # in config file:
    #dataset_name: 'robust04'
    #index_dir: "pyserini-index"
    #collection: "collection.tsv"
    # and dense retrieval config!

    # this could also break it, because it is only trained on generating short msmarco queries, so maybe i should not make it a variable for now
    avg_query_length = {'robust04': 30,
                        'trec-covid': 20}


    config_overwrites_text = ''
    if args.warmstart_model_path:
        if 'large' in args.warmstart_model_path:
            config_overwrites_text = "bert_pretrained_model: t5-large,collection_batch_size: 32,warmstart_model_path: {}, " \
                                 "token_embedder_type: bert_dot, max_length_query_generation: 15".format(args.warmstart_model_path) #, avg_query_length.get(config['dataset_name']))
        else:
            config_overwrites_text = "bert_pretrained_model: t5-small,collection_batch_size: 64,warmstart_model_path: {}, " \
                                     "token_embedder_type: bert_dot, max_length_query_generation: 15".format(
                args.warmstart_model_path)  # , avg_query_length.get(config['dataset_name']))


    print('use following model {}'.format(args.warmstart_model_path))
    print('for the dataset {}'.format(config['dataset_name']))


    #
    # Query Generation
    #

    # add to config:
    #num_queries_per_seq: 5

    print(args.config_file[0])


    if args.query_path is None:
        if args.beam is None:
            print('start with query generation with model')
            run_folder = execute_and_return_run_folder(["python", "matchmaker/autolabel/generate_queries_t5.py", "--config",
                                                        args.config_file[0],
                                                        "--config-overwrites",
                                                        config_overwrites_text,
                                                        "--run-name", "gen_{}_{}".format(config['dataset_name'], args.run_name)])
            print('finished query generation')
            query_path = os.path.join(run_folder, "generated-queries.tsv")
        else:
            print('start with beam query generation with model')
            run_folder = execute_and_return_run_folder(
                ["python", "matchmaker/autolabel/generate_queries_t5_beam.py", "--config",
                 args.config_file[0],
                 "--config-overwrites",
                 config_overwrites_text,
                 "--run-name", "gen_{}_{}".format(config['dataset_name'], args.run_name)])
            print('finished beam query generation')
            query_path = os.path.join(run_folder, "generated-queries.tsv")

    else:
        query_path = args.query_path
        run_folder = '/'.join(query_path.split('/')[:-1])

        print('this is the run folder: {}'.format(run_folder))

    #
    # Pyserini BM25 retrieval for queries for hard negatives
    #
    if args.candidate_path is None:
        print('start pyserini retrieval for generated queries')
        execute_and_return_run_folder(["python", "matchmaker/autolabel_domain/pyserini_bm25_search.py", "--candidate-file-out",
                                                    run_folder,
                                                    "--index-dir",
                                                    config['index_dir'],
                                                    "--query-file", query_path])
        print('finished with bm25 retrieval')
        candidate_file_path = os.path.join(run_folder, 'candidate_file_top100.txt')
    else:
        print('no pyserini search, candidate file is {}'.format(args.candidate_path))
        candidate_file_path = args.candidate_path

    #
    # creation of training triples with bm25 negatives and positives are the documents for which the query was created
    #

    if args.triples_file is not None:
        train_triples_path = args.triples_file
    else:
        print('create triples')

        train_triples_path = os.path.join(run_folder, 'train_triples_bm25neg.tsv')
        execute_and_return_run_folder(["python", "matchmaker/autolabel_domain/autolabel_generate_triples.py", "--out-file",
                                                    train_triples_path,
                                                    "--out-file-ids",
                                                    os.path.join(run_folder, 'train_triples_bm25neg_ids.tsv'),
                                                    "--candidate-file", candidate_file_path,
                                                    "--collection-file", config['collection_tsv'],
                                                    "--query-file", query_path,
                                                    "--candidate-neg", 'bm25_neg',
                                                    "--rank-bound", '50'])


    #
    # score the triples with the teacher!
    #
    # create triples with text
    if args.scoring_model is not None:
        trained_model_path = args.scoring_model
    elif args.machine == 'tr':
        trained_model_path = "/newstorage5/salthamm/msmarco/models/2019-08-01_2109_bert_base_cont/"
        #"/newstorage5/salthamm/msmarco/models/2020-12-02_0833_T2dotrerank_distilbert_warmstartT0_cls_bs32_lr1e6"
    elif args.machine == 'dl':
        trained_model_path = "/newstorage5/salthamm/msmarco/models/2019-08-01_2109_bert_base_cont/"
            #"/home/dlmain/salthammer/msmarco/models/2020-12-02_0833_T2dotrerank_distilbert_warmstartT0_cls_bs32_lr1e6"
    else:
        trained_model_path = "/newstorage5/salthamm/msmarco/models/2019-08-01_2109_bert_base_cont/"
            #"/home/ubuntu/msmarco/models/2020-12-02_0833_T2dotrerank_distilbert_warmstartT0_cls_bs32_lr1e6"

    config_overtext = "train_tsv: {}, trained_model: {}, expirement_base_path: {}".format(train_triples_path, trained_model_path, run_folder)

    print('starting to score the triples with model {}'.format(trained_model_path))

    if args.scores_file is not None:
        scores_file = args.scores_file
    else:
        if 'colbert' in trained_model_path:
            run_folder_scores = execute_and_return_run_folder(
                ["python", "matchmaker/distillation/teacher-train-scorer.py", "--config-file",
                 "config/train/bertcat_scorer.yaml",
                 "config/train/models/colbert.yaml",
                 args.config_file[0],
                 "--config-overwrites",
                 config_overtext,
                 "--run-name", "scores_{}_{}".format(config['dataset_name'], args.run_name)])
        else:
            run_folder_scores = execute_and_return_run_folder(["python", "matchmaker/distillation/teacher-train-scorer.py", "--config-file",
                                                        "config/train/bertcat_scorer.yaml",
                                                        "config/train/models/bert_cat.yaml",
                                                        args.config_file[0],
                                                        "--config-overwrites",
                                                        config_overtext,
                                                        "--run-name", "scores_{}_{}".format(config['dataset_name'], args.run_name)])
        scores_file = os.path.join(run_folder_scores, 'train_scores.tsv')

        print('finished with scoring the triples')

    #
    # filter according to the scores and according to the bm25 top 100
    #

    # 1. filtering: if pos_doc not in the bm25top100 then delete the sample
    # 2. filter_triples.py: if negative score is higher than the positive one then delete
    # 3. filtering: if i have multiple queries per document, then choose the query which has the highest positive score for the document!

    print('start filtering the scored triples')

    execute_and_return_run_folder(["python", "matchmaker/autolabel_domain/filter_triples_scores.py",
                                                "--in-triples",
                                                scores_file,
                                                '--out-triples',
                                                os.path.join(run_folder, 'triples_filtered.tsv'),
                                                '--out-triples-wo-scores',
                                                os.path.join(run_folder, 'triples_filtered_wo.scores.tsv'),
                                                '--filter-multiple-queries',
                                                config.get('filter_multiple_queries', 'True'),
                                                #'--candidate-file',
                                                #candidate_file_path,
                                                '--collection-file',
                                                config['collection_tsv'],
                                                '--query-file',
                                                query_path])

    #
    # train the dr model and include dense retrieval config!
    #

    #run_folder_dr = execute_and_return_run_folder(["python", "matchmaker/train.py", "--config",
    #                                            "config/train/train_dr_adapt.yaml",
    #                                            args.config_file[0],
    #                                            "config/train/models/bert_dot.yaml",
    #                                            "--run-name", "dr_{}_{}".format(config['dataset_name'], args.run_name)])







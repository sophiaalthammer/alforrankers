# Matchmaker Dense Retrieval
# -------------------------------
# Conduct 3 phases of dense retrieval: encoding, indexing, search
# Only in batch form, not really meant for production use of a search engine
#
# - Needs a trained model (via train.py)
# - Measures efficiency & effectiveness on 1 collection + multiple query sets (start a new experiment for another collection)
# - Allows to start new experiment from each of the 3 steps via modes:
#
#         mode                     config-requirement
#
#      1) encode+index+search   -> trained_model folder path
#      2) index+search          -> continue_folder folder path pointing to an experiment started with 1)
#      3) search                -> continue_folder folder path pointing to an experiment started with 2)
#
# - We can do a lot of hyperparameter studies starting from each step, or just run through a full pass once

import argparse
import copy
import os
import glob
from timeit import default_timer
import sys

sys.path.append(os.getcwd())
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # needed because of the scann library
from transformers import logging

logging.set_verbosity_warning()

import torch
import numpy
import random

from allennlp.nn.util import move_to_device

from matchmaker.models.all import get_model, get_word_embedder, build_model
from matchmaker.modules.indexing_heads import *

from matchmaker.utils.utils import *
from matchmaker.utils.config import *
from matchmaker.utils.input_pipeline import allennlp_single_sequence_loader
from matchmaker.utils.performance_monitor import *

from matchmaker.eval import *
from matchmaker.utils.core_metrics import *
from matchmaker.retrieval.faiss_indices import *

from rich.console import Console
from rich.live import Live

console = Console()

MODE_ALL = "encode+index+search"
MODE_START_INDEX = "index+search"
MODE_START_SEARCH = "search"

if __name__ == "__main__":

    #
    # config & mode selection
    # -------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='One of: ' + MODE_ALL + ', ' + MODE_START_INDEX + ', ' + MODE_START_SEARCH)

    parser_rn = parser.add_mutually_exclusive_group(required=True)
    parser_rn.add_argument('--run-name', action='store', dest='run_name',
                           help='run name, used for the run folder (no spaces, special characters), gets timestamp added automatically')
    parser_rn.add_argument('--run-name-fixed', action='store', dest='run_name_fixed',
                           help='fixed run name, used for the run folder (no spaces, special characters)')

    parser.add_argument('--config', nargs='+', action='store', dest='config_file',
                        help='config file with all hyper-params & paths', required=True)
    parser.add_argument('--config-overwrites', action='store', dest='config_overwrites',
                        help='overwrite config values format; (non-)whitespace important! -> key1: value1,key2: value2',
                        required=False)

    args = parser.parse_args()
    config = get_config(args.config_file, args.config_overwrites)
    run_folder = prepare_experiment(args, config)

    if args.mode == MODE_ALL:
        encode_config = config
        index_config = config
        model_config = get_config_single(config["trained_model"])
        print_hello({**model_config, **config}, run_folder, "[Dense Retrieval] Encode & Index & Search",
                    show_settings=["Model", "Trained Checkpoint", "Index", "Collection Batch Size", "Query Batch Size",
                                   "Use ONNX Runtime"])

    elif args.mode == MODE_START_INDEX:
        if "continue_folder" not in config: raise Exception("continue_folder must be set in config")

        encode_folder = config["continue_folder"]
        encode_config = get_config_single(encode_folder)
        model_config = get_config_single(encode_config["trained_model"])
        index_config = config
        print_hello({**model_config, **config, **{"trained_model": encode_config["trained_model"]}}, run_folder,
                    "[Dense Retrieval] Index & Search",
                    show_settings=["Model", "Trained Checkpoint", "Index", "Query Batch Size", "Use ONNX Runtime"])

    elif args.mode == MODE_START_SEARCH:
        if "continue_folder" not in config: raise Exception("continue_folder must be set in config")

        index_folder = config["continue_folder"]
        index_config = get_config_single(index_folder)
        encode_folder = index_config["continue_folder"] if "continue_folder" in index_config else index_folder
        encode_config = get_config_single(encode_folder)
        model_config = get_config_single(encode_config["trained_model"])
        print_hello({**model_config, **config, **{"trained_model": encode_config["trained_model"]}}, run_folder,
                    "[Dense Retrieval] Search",
                    show_settings=["Model", "Trained Checkpoint", "Index", "Query Batch Size", "Use ONNX Runtime"])

    else:
        raise Exception("mode not supported")

    use_onnx = config["onnx_use_inference"]
    if use_onnx:  # prevent errors if onnx is not properly installed (see readme for setup ionstructions)
        import onnxruntime
        from matchmaker.utils.onnx_helper import *

    logger = get_logger_to_file(run_folder, "main")
    logger.info("Running: %s", str(sys.argv))

    torch.manual_seed(model_config["random_seed"])
    numpy.random.seed(model_config["random_seed"])
    random.seed(model_config["random_seed"])
    logger.info("Torch seed: %i ", torch.initial_seed())

    # hardcode gpu usage
    cuda_device = 0  # main cuda device
    perf_monitor = PerformanceMonitor.get()
    perf_monitor.start_block("startup")

    #
    # create and load model instance
    # -------------------------------

    word_embedder, padding_idx = get_word_embedder(model_config)
    model, encoder_type = get_model(model_config, word_embedder, padding_idx)
    model = build_model(model, encoder_type, word_embedder, model_config)

    if model_config.get("model_checkpoint_from_huggingface", False):
        model.from_pretrained(encode_config["trained_model"])
        console.log("[Startup]", "Trained model loaded from huggingface")
    else:
        model_path = os.path.join(encode_config["trained_model"], "best-model.pytorch-state-dict")
        load_result = model.load_state_dict(torch.load(model_path), strict=False)
        logger.info('Warmstart init model from:  %s', model_path)
        logger.info(load_result)
        console.log("[Startup]", "Trained model loaded locally; result:", load_result)

    vector_dimensions = model.get_output_dim()

    #
    # setup heads wrapping the model for indexing & searching
    #
    if args.mode == MODE_ALL:
        model_indexer = CollectionIndexerHead(model, use_fp16=False if use_onnx else model_config["use_fp16"]).cuda()
        model_indexer.eval()

        if use_onnx:
            console.log("[Startup]", "Using ONNX, converting & optimizing indexer ... ")
            convert_and_optimize(model_indexer, os.path.join(run_folder, "indexer-model.onnx"),
                                 model_config["use_fp16"])
            del model_indexer
            onnx_indexer = onnxruntime.InferenceSession(os.path.join(run_folder, "indexer-model.onnx"),
                                                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    model_searcher = QuerySearcherHead(model, use_fp16=False if use_onnx else model_config["use_fp16"]).cuda()
    model_searcher.eval()

    if use_onnx:
        console.log("[Startup]", "Using ONNX, converting & optimizing searcher ... ")
        convert_and_optimize(model_searcher, os.path.join(run_folder, "searcher-model.onnx"), model_config["use_fp16"])
        del model_searcher

    logger.info('Model %s total parameters: %s', model_config["model"],
                sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info('Network: %s', model)
    del model

    #
    # setup multi-gpu, todo fix to forward only calls
    #
    if torch.cuda.device_count() > 1 and not use_onnx:
        console.log("Let's use", torch.cuda.device_count(), "GPUs!")
        if args.mode == MODE_ALL:
            model_indexer = torch.nn.DataParallel(model_indexer)
            model_indexer.eval()
        if config["query_batch_size"] > 1:
            model_searcher = torch.nn.DataParallel(model_searcher)
            model_searcher.eval()
        else:
            console.log("[Startup] Info: Using single GPU for search, as batch_size is set to 1")
    elif torch.cuda.device_count() > 1 and use_onnx:
        console.log("[Startup] Info: ONNX currently only uses 1 GPU")

    perf_monitor.set_gpu_info(torch.cuda.device_count(), torch.cuda.get_device_name())
    perf_monitor.stop_block("startup")

    print(config["faiss_ivf_list_count"])

    try:

        #
        # 1) Encode
        # -------------------------------

        token_base_size = config["token_block_size"]

        if args.mode == MODE_ALL:

            console.log("[Encoding]", "Encoding collection from: ", config["cluster_queries_tsv"])

            doc_infos = {}
            seq_ids = []

            token_base_number = 0
            token_base = np.memmap(os.path.join(run_folder, "token_reps_" + str(token_base_number) + ".npy"),
                                   dtype=numpy.dtype(config["token_dtype"]),
                                   mode="w+", shape=(token_base_size, vector_dimensions))

            current_ids = np.ndarray(shape=(token_base_size), dtype='int64')
            id_mapping = []
            token_insert_index = 0
            storage = []
            storage_filled_to_index = []

            input_loader = allennlp_single_sequence_loader(model_config, config, config["cluster_queries_tsv"],
                                                           sequence_type="doc")
            perf_monitor.start_block("encode")
            start_time = default_timer()
            # import pprofile
            # prof = pprofile.Profile()

            with torch.no_grad(), Live("[bold magenta]           Loading...", console=console,
                                       auto_refresh=False) as status:  # ,prof():
                batch_number = 0
                sequence_number = 0

                for batch in input_loader:

                    if use_onnx:
                        output = onnx_indexer.run(None, {'input_ids': batch["seq_tokens"]['input_ids'].numpy(),
                                                         'attention_mask': batch["seq_tokens"]['attention_mask'].to(
                                                             torch.int64).numpy()})[0]
                    else:
                        batch = move_to_device(copy.deepcopy(batch), cuda_device)
                        output = model_indexer.forward(batch["seq_tokens"])
                        output = output.cpu().numpy()  # get the output back to the cpu - in one piece

                    # compare ONNX Runtime and PyTorch results
                    # np.testing.assert_allclose(output, ort_outs[0], rtol=1e-03, atol=1e-05)

                    for sample_i, seq_id in enumerate(batch["seq_id"]):  # operate on cpu memory

                        # assuming either 1 vec or 1-n
                        # if 1-n remove 0 vectors as padding (colbert,tk,tkl models)
                        current_reps = output[sample_i]
                        dim_count = len(current_reps.shape)
                        if dim_count == 2:
                            current_reps = current_reps[np.abs(current_reps).sum(-1) > 0,
                                           :]  # a bit dicey, but i guess we are fine because what is the prob. of something being 0 without being padding

                        vec_count = 1 if dim_count == 1 else current_reps.shape[0]

                        if token_insert_index + vec_count > token_base_size:
                            storage.append(token_base[:token_insert_index])
                            id_mapping.append(current_ids[:token_insert_index])
                            current_ids = np.ndarray(shape=(token_base_size), dtype='int64')
                            storage_filled_to_index.append(token_insert_index)

                            token_base_number += 1
                            token_insert_index = 0
                            token_base = np.memmap(
                                os.path.join(run_folder, "token_reps_" + str(token_base_number) + ".npy"),
                                dtype=numpy.dtype(config["token_dtype"]),
                                mode="w+", shape=(token_base_size, vector_dimensions))

                        start_index = token_insert_index
                        token_insert_index = token_insert_index + vec_count
                        token_base[start_index:token_insert_index] = current_reps
                        current_ids[start_index:token_insert_index] = len(seq_ids)

                        doc_infos[seq_id] = (token_base_number, start_index, token_insert_index)
                        seq_ids.append(seq_id)

                    batch_number += 1
                    sequence_number += len(batch["seq_id"])
                    if batch_number % 10 == 0: status.update("[bold magenta]           Progress ... Batch No.: " + str(
                        batch_number) + " | Sequence No.: " + str(sequence_number) + " | Seq. / second: " + \
                                                             "{:.2f}".format(
                                                                 sequence_number / (default_timer() - start_time)),
                                                             refresh=True)

            # prof.print_stats()

            # save last token reps
            storage.append(token_base[:token_insert_index])
            id_mapping.append(current_ids[:token_insert_index])
            storage_filled_to_index.append(token_insert_index)

            saveCompressed(os.path.join(run_folder, "doc_infos.npz"), doc_infos=doc_infos, id_mapping=id_mapping,
                           seq_ids=seq_ids, storage_filled_to_index=storage_filled_to_index)
            if not use_onnx:
                perf_monitor.log_unique_value("encoding_gpu_mem",
                                              str(torch.cuda.memory_allocated() / float(1e9)) + " GB")
                perf_monitor.log_unique_value("encoding_gpu_mem_max",
                                              str(torch.cuda.max_memory_allocated() / float(1e9)) + " GB")
            perf_monitor.log_unique_value("encoded_size_on_disk", str(
                sum(os.path.getsize(f) for f in glob.glob(os.path.join(run_folder, "token_reps_*"))) / float(
                    1e9)) + " GB")

            perf_monitor.stop_block("encode", len(seq_ids))

        #
        # skip encoding
        #
        else:
            console.log("[Encoding]", "Skipping encoding; loading collection vectors & info from: ", encode_folder)
            dfs = numpy.load(os.path.join(encode_folder, "doc_infos.npz"), allow_pickle=True)
            doc_infos = dfs.get("doc_infos")[()]
            id_mapping = dfs.get("id_mapping")[()]
            seq_ids = dfs.get("seq_ids")[()]
            storage_filled_to_index = dfs.get("storage_filled_to_index")[()]

            storage = []
            for f in range(0, len(glob.glob(os.path.join(encode_folder, "token_reps_*")))):
                storage.append(np.memmap(os.path.join(encode_folder, "token_reps_" + str(f) + ".npy"),
                                         dtype=numpy.dtype(encode_config["token_dtype"]),
                                         mode="r", shape=(token_base_size, vector_dimensions))[
                               :storage_filled_to_index[f]])

        #
        # 2) Nearest neighbor indexing
        # -------------------------

        # if index_config["faiss_index_type"] == "ondisk_sharding":
        #     indexer = FaissShardedOnDiskIdIndexer(index_config, vector_dimensions)
        # elif index_config["faiss_index_type"] == "full":
        #     indexer = FaissIdIndexer(index_config, vector_dimensions)
        # elif index_config["faiss_index_type"] == "ivf":
        #     indexer = FaissIVFIndexer(index_config, vector_dimensions)  # todo
        # elif index_config["faiss_index_type"] == "hnsw":
        #     indexer = FaissHNSWIndexer(index_config, vector_dimensions)
        # elif index_config["faiss_index_type"] == "scann":
        #     from matchmaker.retrieval.scann_index import ScaNNIndexer  # import here, because it only works on linux
        #
        #     indexer = ScaNNIndexer(index_config, vector_dimensions)
        # else:
        #     raise Exception("faiss_index_type not supported")
        #
        # # we don't save the full index, but rebuilt it every time (just loading the vectors basically)
        # if args.mode != MODE_START_SEARCH or index_config["faiss_index_type"] == "full":

        perf_monitor.start_block("indexing")

        indexer = FaissDynamicIndexer(config, vector_dimensions)

        #indexer.prepare([storage])
        #indexer.index_all([id_mapping], [storage])

        indexer.prepare(storage)
        indexer.index_all(id_mapping, storage)

        #
        # cluster info output
        # -------------------------
        perf_monitor.start_block("output")

        id_text = {}
        with open(config["cluster_queries_tsv"], "r", encoding="utf8") as qf:
            for l in qf:
                l = l.split("\t")
                id_text[l[0]] = l[1].strip()

        clusters = [[] for _ in range(config["faiss_ivf_list_count"])]

        input_loader = allennlp_single_sequence_loader(model_config, config, config["cluster_queries_tsv"],
                                                       sequence_type="query", force_exact_batch_size=True)


        with torch.no_grad(), Live("[bold magenta]           Loading...", console=console,
                                   auto_refresh=False) as status:
            i = 0

            for batch_orig in input_loader:

                #if batch_orig == None:
                #    break

                batch_size = len(batch_orig["seq_id"])

                perf_monitor.start_block("search_query_encode")

                batch = move_to_device(copy.deepcopy(batch_orig), cuda_device)
                output = model_searcher.forward(batch["seq_tokens"], search_type="encode")
                output = output.cpu().numpy()  # get the output back to the cpu - in one piece

                for sample_i, seq_id in enumerate(batch_orig["seq_id"]):
                    _, _, centroid_ids = indexer.search_single(output[sample_i], 1)

                    clusters[int(centroid_ids)].append(seq_id)

                #    id_mapping.append(len(seq_ids))
                #    seq_ids.append(seq_id)
                # storage.append(output)

        with open(os.path.join(run_folder, "cluster-assignment-ids.tsv"), "w", encoding="utf8") as out_file, \
                open(os.path.join(run_folder, "cluster-assignment-text.tsv"), "w",
                     encoding="utf8") as out_file_text:
            for clust in clusters:
                out_file.write("\t".join(idx for idx in clust) + "\n")
                out_file_text.write("\n".join(idx + "\t" + id_text[idx] for idx in clust) + \
                                    "\n--------------------------------------------\n")

        perf_monitor.stop_block("output")
        perf_monitor.save_summary(os.path.join(run_folder, "perf-monitor.txt"))


    except KeyboardInterrupt:
        logger.info('-' * 20)
        logger.info('Manual Stop!')
        console.log("Manual Stop! Bye :)")

    except Exception as e:
        logger.info('-' * 20)
        logger.exception('[train] Got exception: ')
        logger.info('Exiting from training early')
        console.log("[red]Exception! ", str(e))
        console.print_exception()

        exit(1)

    finally:

        # cleanup the onnx model, this will come back to bite me some day, but for now let's save storage space!
        if os.path.exists(os.path.join(run_folder, "indexer-model.onnx")):
            os.remove(os.path.join(run_folder, "indexer-model.onnx"))
        if os.path.exists(os.path.join(run_folder, "searcher-model.onnx")):
            os.remove(os.path.join(run_folder, "searcher-model.onnx"))

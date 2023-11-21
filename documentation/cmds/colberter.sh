token_dim=32
CUDA_VISIBLE_DEVICES=0 python matchmaker/train.py --run-name colberter-v3_PL_dim128-$token_dim \
--config-file config/train/defaults.yaml config/train/data/id-msmarco-passage-dot-rank.yaml config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml \
--config-overwrites "colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,colberter_aggregate_unique_ids: False,warmstart_model_path: /newstorage5/shofstae/msmarco-passage-experiments/2021-09-15_1254_colbert_dim768_bs32_PL_3bert/best-model.pytorch-state-dict"


token_dim=32
CUDA_VISIBLE_DEVICES=5 python matchmaker/train.py --run-name "colberter-v3_PL_dim128-"$token_dim"_weightedscore1_0" \
--config-file config/train/defaults.yaml config/train/data/tr-msmarco-passage-dot-rank.yaml config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml \
--config-overwrites "secondary_loss_lambda: 1,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,colberter_aggregate_unique_ids: False,warmstart_model_path: /newstorage5/shofstae/msmarco-passage-experiments/2021-09-15_1254_colbert_dim768_bs32_PL_3bert/best-model.pytorch-state-dict"

token_dim=64
CUDA_VISIBLE_DEVICES=4 python matchmaker/train.py --run-name "colberter-v3_PL_dim128-"$token_dim"_weightedscore0_2" \
--config-file config/train/defaults.yaml config/train/data/tr-msmarco-passage-dot-rank.yaml config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml \
--config-overwrites "secondary_loss_lambda: 0.2,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,colberter_aggregate_unique_ids: False,warmstart_model_path: /newstorage5/shofstae/msmarco-passage-experiments/2021-09-15_1254_colbert_dim768_bs32_PL_3bert/best-model.pytorch-state-dict"


token_dim=32
CUDA_VISIBLE_DEVICES=0 python matchmaker/train.py --run-name "colberter-v3_PL_dim128-"$token_dim"_weightedscore0_5" \
--config-file config/train/defaults.yaml config/train/data/dl-msmarco-passage-dot-rank.yaml config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml \
--config-overwrites "secondary_loss_lambda: 0.5,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,colberter_aggregate_unique_ids: False,warmstart_model_path: /scratch/hofstaetter/experiments/msmarco-passage-v1/2021-09-15_1254_colbert_dim768_bs32_PL_3bert/best-model.pytorch-state-dict"

token_dim=32
CUDA_VISIBLE_DEVICES=1 python matchmaker/train.py --run-name "colberter-v3_PL_dim128-"$token_dim"_tokenonlyscore0_5" \
--config-file config/train/defaults.yaml config/train/data/dl-msmarco-passage-dot-rank.yaml config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml \
--config-overwrites "secondary_loss_lambda: 0.5,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,colberter_aggregate_unique_ids: False,warmstart_model_path: /scratch/hofstaetter/experiments/msmarco-passage-v1/2021-09-15_1254_colbert_dim768_bs32_PL_3bert/best-model.pytorch-state-dict"

token_dim=32
CUDA_VISIBLE_DEVICES=0 python matchmaker/train.py --run-name "colberter-v3_PL_dim128-"$token_dim"_weightedscore0_1" \
--config-file config/train/defaults.yaml config/train/data/id-msmarco-passage-dot-rank.yaml config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml \
--config-overwrites "secondary_loss_lambda: 0.1,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,colberter_aggregate_unique_ids: False,warmstart_model_path: /newstorage5/shofstae/msmarco-passage-experiments/2021-09-15_1254_colbert_dim768_bs32_PL_3bert/best-model.pytorch-state-dict"

token_dim=32
CUDA_VISIBLE_DEVICES=1 python matchmaker/train.py --run-name "colberter-v3_PL_dim128-"$token_dim"_weightedscore0_2" \
--config-file config/train/defaults.yaml config/train/data/id-msmarco-passage-dot-rank.yaml config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml \
--config-overwrites "secondary_loss_lambda: 0.2,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,colberter_aggregate_unique_ids: False,warmstart_model_path: /newstorage5/shofstae/msmarco-passage-experiments/2021-09-15_1254_colbert_dim768_bs32_PL_3bert/best-model.pytorch-state-dict"



token_dim=8
token_dim=16
token_dim=32
token_dim=64

server_config="config/train/data/dl-msmarco-passage-dot-rank.yaml"
# stage 1 -> dimred
server_warmstart="/scratch/hofstaetter/experiments/msmarco-passage-v1/2021-09-15_1254_colbert_dim768_bs32_PL_3bert/best-model.pytorch-state-dict"
# stage 2 -> dimred + bow
server_warmstart="/scratch/hofstaetter/experiments/msmarco-passage-v1/2022-01-06_1901_colberter-v3_PL_dim128-32_weightedscore0_5/best-model.pytorch-state-dict"
server_warmstart="/scratch/hofstaetter/experiments/msmarco-passage-v1/2022-01-08_1740_colberter-v3_PL_dim128-64_weightedscore0_5/best-model.pytorch-state-dict"
# stage 3 -> dimred + bow + stop
server_warmstart="/scratch/hofstaetter/experiments/msmarco-passage-v1/2022-01-08_1738_colberter-v3_PL_dim128-32_weightedscore0_5_unique-bow/best-model.pytorch-state-dict"
server_warmstart="/scratch/hofstaetter/experiments/msmarco-passage-v1/2022-01-12_1543_colberter-v3_PL_dim128-64_weightedscore0_5_unique-bow-stemmed/best-model.pytorch-state-dict"

server_warmstart="/scratch/hofstaetter/experiments/msmarco-passage-v1/2022-01-16_1303_colberter-v3_PL_dim128-32_weightedscore0_1_unique-bow-stemmed_stop0_75/best-model.pytorch-state-dict"
server_warmstart="/scratch/hofstaetter/experiments/msmarco-passage-v1/2022-01-14_1749_colberter-v3_PL_dim128-32_weightedscore0_1_unique-bow-stemmed/best-model.pytorch-state-dict"

server_config="config/train/data/id-msmarco-passage-dot-rank.yaml"
server_config="config/train/data/tr-msmarco-passage-dot-rank.yaml"

# stage 1 -> dimred
server_warmstart="/newstorage5/shofstae/msmarco-passage-experiments/2021-09-15_1254_colbert_dim768_bs32_PL_3bert/best-model.pytorch-state-dict"
# stage 2 -> dimred + bow
server_warmstart="/newstorage5/shofstae/msmarco-passage-experiments/2022-01-08_1740_colberter-v3_PL_dim128-64_weightedscore0_5/best-model.pytorch-state-dict"
server_warmstart="/newstorage5/shofstae/msmarco-passage-experiments/2022-01-06_1901_colberter-v3_PL_dim128-32_weightedscore0_5/best-model.pytorch-state-dict"
server_warmstart="/newstorage5/shofstae/msmarco-passage-experiments/2022-01-08_1747_colberter-v3_PL_dim128-16_weightedscore0_5/best-model.pytorch-state-dict"
server_warmstart="/newstorage5/shofstae/msmarco-passage-experiments/2022-01-08_1746_colberter-v3_PL_dim128-8_weightedscore0_5/best-model.pytorch-state-dict"
# stage 3 -> dimred + bow + stop

# mini experiment
server_warmstart="/newstorage5/shofstae/msmarco-passage-experiments/2022-01-08_1738_colberter-v3_PL_dim128-32_weightedscore0_5_unique-bow/best-model.pytorch-state-dict"
server_warmstart="/newstorage5/shofstae/msmarco-passage-experiments/2022-01-12_1504_colberter-v3_PL_dim128-64_weightedscore0_5_unique-bow-stemmed_stop1/best-model.pytorch-state-dict"
server_warmstart="/newstorage5/shofstae/msmarco-passage-experiments/2022-01-16_1303_colberter-v3_PL_dim128-32_weightedscore0_1_unique-bow-stemmed_stop0_75/best-model.pytorch-state-dict"

server_warmstart="/newstorage5/shofstae/msmarco-passage-experiments/2022-01-14_1749_colberter-v3_PL_dim128-32_weightedscore0_1_unique-bow-stemmed/best-model.pytorch-state-dict"
server_warmstart="/newstorage5/shofstae/msmarco-passage-experiments/2022-01-06_1901_colberter-v3_PL_dim128-32_weightedscore0_5/best-model.pytorch-state-dict"

cuda_id=5
cuda_id=1 

# stage 1 -> dimred
CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/train.py --run-name "colberter-v3_PL_dim128-"$token_dim"_weightedscore0_5" \
--config-file config/train/defaults.yaml $server_config config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml \
--config-overwrites "secondary_loss_lambda: 0.5,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,colberter_aggregate_unique_ids: False,warmstart_model_path: $server_warmstart"

# stage 2 -> dimred + bow
CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/train.py --run-name "colberter-v3_PL_dim128-"$token_dim"_weightedscore0_1_unique-bow-stemmed" \
--config-file config/train/defaults.yaml $server_config config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml \
--config-overwrites "secondary_loss_lambda: 0.1,colberter_aggregate_unique_ids: True,colberter_aggregate_unique_ids_type: stemmed,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,warmstart_model_path: $server_warmstart"

# stage 3 -> dimred + bow + stop
CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/train.py --run-name "colberter-v3_PL_dim128-"$token_dim"_weightedscore0_1_unique-bow-stemmed_stop0_75" \
--config-file config/train/defaults.yaml $server_config config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml config/train/modes/sparsity.yaml \
--config-overwrites "colberter_use_contextualized_stopwords: True,sparsity_loss_lambda_factor: 0.75,secondary_loss_lambda: 0.1,colberter_aggregate_unique_ids: True,colberter_aggregate_unique_ids_type: stemmed,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,warmstart_model_path: $server_warmstart"


# stage mini 1
CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/train.py --run-name "colberter-exact_mini-v3_PL_dim128-"$token_dim"-1_weightedscore0_5" \
--config-file config/train/defaults.yaml $server_config config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml \
--config-overwrites "colberter_compress_to_exact_mini_mode: True,colberter_second_compress_dim: 1,secondary_loss_lambda: 0.5,colberter_aggregate_unique_ids: False,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,warmstart_model_path: $server_warmstart"

# stage mini 2
CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/train.py --run-name "colberter-exact_mini-v3_PL_dim128-"$token_dim"-2_weightedscore0_5_unique-bow" \
--config-file config/train/defaults.yaml $server_config config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml \
--config-overwrites "colberter_compress_to_exact_mini_mode: True,secondary_loss_lambda: 0.5,colberter_aggregate_unique_ids: True,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,warmstart_model_path: $server_warmstart"

CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/train.py --run-name "colberter-exact_mini-v3_PL_dim128-"$token_dim"-"$token_dim_mini"_weightedscore0_1_unique-bow-stemmed-stop0_75" \
--config-file config/train/defaults.yaml $server_config config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml  config/train/modes/sparsity.yaml \
--config-overwrites "colberter_use_contextualized_stopwords: True,sparsity_loss_lambda_factor: 0.75,colberter_compress_to_exact_mini_mode: True,colberter_second_compress_dim: $token_dim_mini,secondary_loss_lambda: 0.1,colberter_aggregate_unique_ids: True,colberter_aggregate_unique_ids_type: stemmed,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,warmstart_model_path: $server_warmstart"


CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/train.py --run-name "colberter-exact_mini-v3_PL_dim128-"$token_dim"-"$token_dim_mini"_weightedscore0_1_unique-bow-stemmed" \
--config-file config/train/defaults.yaml $server_config config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml  \
--config-overwrites "colberter_use_contextualized_stopwords: False,colberter_compress_to_exact_mini_mode: True,colberter_second_compress_dim: $token_dim_mini,secondary_loss_lambda: 0.1,colberter_aggregate_unique_ids: True,colberter_aggregate_unique_ids_type: stemmed,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,warmstart_model_path: $server_warmstart"

CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/train.py --run-name "colberter-exact_mini-v3_PL_dim128-"$token_dim"-"$token_dim_mini"_weightedscore0_1" \
--config-file config/train/defaults.yaml $server_config config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml  \
--config-overwrites "colberter_use_contextualized_stopwords: False,colberter_compress_to_exact_mini_mode: True,colberter_second_compress_dim: $token_dim_mini,secondary_loss_lambda: 0.1,colberter_aggregate_unique_ids: False,colberter_aggregate_unique_ids_type: stemmed,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,warmstart_model_path: $server_warmstart"


CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/train.py --run-name "colberter-mini-v3_PL_dim128-"$token_dim"-"$token_dim_mini"_weightedscore0_1" \
--config-file config/train/defaults.yaml $server_config config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml  \
--config-overwrites "colberter_use_contextualized_stopwords: False,colberter_compress_to_exact_mini_mode: True,colberter_second_compress_dim: $token_dim_mini,secondary_loss_lambda: 0.1,colberter_aggregate_unique_ids: False,colberter_aggregate_unique_ids_type: stemmed,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,warmstart_model_path: $server_warmstart"


# stage mini 3
token_dim_mini=16



CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/train.py --run-name "colberter-exact_mini-v3_PL_dim128-"$token_dim"-"$token_dim_mini"_weightedscore0_5_unique-bow-stemmed-stop1" \
--config-file config/train/defaults.yaml $server_config config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml config/train/modes/sparsity.yaml \
--config-overwrites "colberter_use_contextualized_stopwords: True,sparsity_loss_lambda_factor: 1,colberter_compress_to_exact_mini_mode: True,colberter_second_compress_dim: $token_dim_mini,secondary_loss_lambda: 0.5,colberter_aggregate_unique_ids: True,colberter_aggregate_unique_ids_type: stemmed,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,warmstart_model_path: $server_warmstart"

CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/train.py --run-name "colberter-mini-v3_PL_dim128-"$token_dim"-"$token_dim_mini"_weightedscore0_5_unique-bow-stemmed-stop1" \
--config-file config/train/defaults.yaml $server_config config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml config/train/modes/sparsity.yaml \
--config-overwrites "colberter_use_contextualized_stopwords: True,sparsity_loss_lambda_factor: 1,colberter_compress_to_exact_mini_mode: False,colberter_second_compress_dim: $token_dim_mini,secondary_loss_lambda: 0.5,colberter_aggregate_unique_ids: True,colberter_aggregate_unique_ids_type: stemmed,colberter_retrieval_compression_dim: 128,colberter_compression_dim: $token_dim,warmstart_model_path: $server_warmstart"




CUDA_VISIBLE_DEVICES=0 python matchmaker/train.py --run-name colberter-v2_PL_unique-whole-word_stopword4_dim128-32 \
--config-file config/train/defaults.yaml config/train/data/id-msmarco-passage-dot-rank.yaml config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml config/train/modes/sparsity.yaml \
--config-overwrites "colberter_use_contextualized_stopwords: True,sparsity_loss_lambda_factor: 4,colberter_retrieval_compression_dim: 128,colberter_compression_dim: 32,colberter_aggregate_unique_ids: True,warmstart_model_path: /newstorage5/shofstae/msmarco-passage-experiments/2021-12-29_1623_colberter-v2_PL_unique-whole-word_dim128-32/best-model.pytorch-state-dict"


#
# unique whole word comp
#



CUDA_VISIBLE_DEVICES=0 python matchmaker/train.py --run-name colberter-v2_PL_unique-whole-word_dim128-64 \
--config-file config/train/defaults.yaml config/train/data/dl-msmarco-passage-dot-rank.yaml config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml \
--config-overwrites "colberter_retrieval_compression_dim: 128,colberter_compression_dim: 64,colberter_aggregate_unique_ids: True,warmstart_model_path: /scratch/hofstaetter/experiments/msmarco-passage-v1/2022-01-02_1230_colberter-v2_PL_dim128-64/best-model.pytorch-state-dict"

CUDA_VISIBLE_DEVICES=1 python matchmaker/train.py --run-name colberter-v2_PL_unique-whole-word_dim128-8 \
--config-file config/train/defaults.yaml config/train/data/dl-msmarco-passage-dot-rank.yaml config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml \
--config-overwrites "colberter_retrieval_compression_dim: 128,colberter_compression_dim: 8,colberter_aggregate_unique_ids: True,warmstart_model_path: /scratch/hofstaetter/experiments/msmarco-passage-v1/2021-12-31_1023_colberter-v2_PL_dim128-8/best-model.pytorch-state-dict"


CUDA_VISIBLE_DEVICES=1 python matchmaker/train.py --run-name colberter-v2_PL_unique-whole-word_stopword1_5_dim128-8 \
--config-file config/train/defaults.yaml config/train/data/dl-msmarco-passage-dot-rank.yaml config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml config/train/modes/sparsity.yaml \
--config-overwrites "colberter_retrieval_compression_dim: 128,colberter_compression_dim: 8,colberter_aggregate_unique_ids: True,colberter_use_contextualized_stopwords: True,sparsity_loss_lambda_factor: 1.5,warmstart_model_path: /scratch/hofstaetter/experiments/msmarco-passage-v1/2022-01-02_1234_colberter-v2_PL_unique-whole-word_dim128-8/best-model.pytorch-state-dict"


#
# unique whole word - stemmed
#

CUDA_VISIBLE_DEVICES=0 python matchmaker/train.py --run-name colberter-v2_PL_unique-whole-word-stemmed_stopword4_dim128-32 \
--config-file config/train/defaults.yaml config/train/data/id-msmarco-passage-dot-rank.yaml config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml config/train/modes/sparsity.yaml \
--config-overwrites "colberter_retrieval_compression_dim: 128,colberter_compression_dim: 32,colberter_aggregate_unique_ids: True,colberter_aggregate_unique_ids_type: stemmed,\
colberter_use_contextualized_stopwords: True,sparsity_loss_lambda_factor: 4,\
warmstart_model_path: /newstorage5/shofstae/msmarco-passage-experiments/2021-12-29_1625_colberter-v2_PL_unique-whole-word-stemmed_dim128-32/best-model.pytorch-state-dict"

CUDA_VISIBLE_DEVICES=0 python matchmaker/train.py --run-name colberter-v2_PL_unique-whole-word-stemmed_stopword1.5_dim128-16 \
--config-file config/train/defaults.yaml config/train/data/id-msmarco-passage-dot-rank.yaml config/train/models/colberter.yaml config/train/modes/pseudo_labels.yaml config/train/modes/sparsity.yaml \
--config-overwrites "colberter_retrieval_compression_dim: 128,colberter_compression_dim: 16,colberter_aggregate_unique_ids: True,colberter_aggregate_unique_ids_type: stemmed,\
colberter_use_contextualized_stopwords: True,sparsity_loss_lambda_factor: 1.5,\
warmstart_model_path: /newstorage5/shofstae/msmarco-passage-experiments/2021-12-30_1035_colberter-v2_PL_unique-whole-word-stemmed_dim128-16/best-model.pytorch-state-dict"



#
# colberter retrieval 
#

CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py index+search \
--run-name colberter-exact_mini-v3_PL_dim128-32-1_weightedscore0_1_unique-bow-stemmed-stop0_75_qb1 \
--config config/dense_retrieval/id-msmarco-passage-v1.yaml \
--config-overwrites "query_batch_size: 1,continue_folder: /home/shofstae/msmarco-passage-v1-retrieval-experiments/2022-01-21_1015_colberter-exact_mini-v3_PL_dim128-32-1_weightedscore0_1_unique-bow-stemmed-stop0_75"


CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py index+search \
--run-name colberter-mini-v3_PL_dim128-32-16_weightedscore0_1_unique-bow-stemmed-stop0_75_qb1 \
--config config/dense_retrieval/dl-msmarco.yaml \
--config-overwrites "query_batch_size: 1,continue_folder: /scratch/hofstaetter/msmarco-passage-retrieval-experiments/2022-01-20_1629_colberter-mini-v3_PL_dim128-32-16_weightedscore0_1_unique-bow-stemmed-stop0_75"


CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py index+search \
--run-name colberter-exact_mini-v3_PL_dim128-32-1_weightedscore0_1_unique-bow-stemmed-stop0_75 \
--config config/dense_retrieval/id-msmarco-passage-v1.yaml \
--config-overwrites "continue_folder: /home/shofstae/msmarco-passage-v1-retrieval-experiments/2022-01-21_1015_colberter-exact_mini-v3_PL_dim128-32-1_weightedscore0_1_unique-bow-stemmed-stop0_75"

CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
--run-name colberter-v3_PL_dim128-32_weightedscore0_1_unique-bow-stemmed \
--config config/dense_retrieval/id-msmarco-passage-v1.yaml \
--config-overwrites "trained_model: /newstorage5/shofstae/msmarco-passage-experiments/2022-01-14_1749_colberter-v3_PL_dim128-32_weightedscore0_1_unique-bow-stemmed"


CUDA_VISIBLE_DEVICES=0 python matchmaker/colberter_retrieval.py encode+index+search \
--run-name colberter-exact_mini-v3_PL_dim128-64-1_weightedscore0_5_unique-bow-stemmed-stop1 \
--config config/dense_retrieval/tr-msmarco.yaml \
--config-overwrites "trained_model: /newstorage5/shofstae/msmarco-passage-experiments/2022-01-14_1052_colberter-exact_mini-v3_PL_dim128-64-1_weightedscore0_5_unique-bow-stemmed-stop1"


CUDA_VISIBLE_DEVICES=1,2 python matchmaker/colberter_retrieval.py index+search \
--run-name colberter-v2_PL_dim128-32 \
--config /mnt/nvme-local/hofstaetter/msmarco-retrieval-experiments/2022-01-07_0759_colberter-v2_PL_dim128-32/config.yaml \
--config-overwrites "continue_folder: /mnt/nvme-local/hofstaetter/msmarco-retrieval-experiments/2022-01-07_0759_colberter-v2_PL_dim128-32\n\
query_sets:\n\
  trec19+20-combined_top1000:\n\
    queries_tsv: /data01/hofstaetter/data/msmarco-passage/trec19+20-combined-queries.tsv\n\
    qrels: /data01/hofstaetter/data/msmarco-passage/trec19+20-combined-qrels.txt\n\
    binarization_point: 2\n\
    top_n: 1000"



#
# out-of-domain experiments
#
trained_model_path="/newstorage5/shofstae/msmarco-passage-experiments/2022-01-19_1116_colberter-exact_mini-v3_PL_dim128-32-1_weightedscore0_1_unique-bow-stemmed-stop0_75"
trained_model_name="colberter-exact_mini-v3_PL_dim128-32-1_weightedscore0_1_unique-bow-stemmed-stop0_75"

trained_model_path="/newstorage5/shofstae/msmarco-passage-experiments/2022-01-16_1303_colberter-v3_PL_dim128-32_weightedscore0_1_unique-bow-stemmed_stop0_75"
trained_model_name="colberter-v3_PL_dim128-32_weightedscore0_1_unique-bow-stemmed_stop0_75"

time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
--run-name "robust04_"$trained_model_name \
--config config/dense_retrieval/tr-robust04.yaml config/dense_retrieval/tr-out-of-domain.yaml \
--config-overwrites "trained_model: $trained_model_path"
time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
--run-name "beir_trec-covid_"$trained_model_name \
--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/beir/trec-covid/config.yaml \
--config-overwrites "trained_model: $trained_model_path"
time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
--run-name "beir_dbpedia-entity_"$trained_model_name \
--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/beir/dbpedia-entity/config.yaml \
--config-overwrites "trained_model: $trained_model_path"
time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
--run-name "beir_nfcorpus_"$trained_model_name \
--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/beir/nfcorpus/config.yaml \
--config-overwrites "trained_model: $trained_model_path"
time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
--run-name "antique_"$trained_model_name \
--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/antique/config.yaml \
--config-overwrites "trained_model: $trained_model_path"
time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
--run-name "trec_podcast_"$trained_model_name \
--config config/dense_retrieval/tr-podcast.yaml config/dense_retrieval/tr-out-of-domain.yaml  \
--config-overwrites "trained_model: $trained_model_path"
time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
--run-name "tripclick_"$trained_model_name \
--config config/dense_retrieval/tr-tripclick.yaml config/dense_retrieval/tr-out-of-domain.yaml \
--config-overwrites "trained_model: $trained_model_path"



trained_model_path="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
trained_model_name="tasb_dim768_hf"

time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/dense_retrieval.py encode+index+search \
--run-name "robust04_"$trained_model_name \
--config config/dense_retrieval/tr-out-of-domain.yaml config/dense_retrieval/tr-robust04.yaml \
--config-overwrites "trained_model: $trained_model_path"
time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/dense_retrieval.py encode+index+search \
--run-name "beir_trec-covid_"$trained_model_name \
--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/beir/trec-covid/config.yaml \
--config-overwrites "trained_model: $trained_model_path"
time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/dense_retrieval.py encode+index+search \
--run-name "beir_dbpedia-entity_"$trained_model_name \
--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/beir/dbpedia-entity/config.yaml \
--config-overwrites "trained_model: $trained_model_path"
time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/dense_retrieval.py encode+index+search \
--run-name "beir_nfcorpus_"$trained_model_name \
--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/beir/nfcorpus/config.yaml \
--config-overwrites "trained_model: $trained_model_path"
time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/dense_retrieval.py encode+index+search \
--run-name "antique_"$trained_model_name \
--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/antique/config.yaml \
--config-overwrites "trained_model: $trained_model_path"
time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/dense_retrieval.py encode+index+search \
--run-name "trec_podcast_"$trained_model_name \
--config config/dense_retrieval/tr-podcast.yaml config/dense_retrieval/tr-out-of-domain.yaml  \
--config-overwrites "trained_model: $trained_model_path"
time CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/dense_retrieval.py encode+index+search \
--run-name "tripclick_"$trained_model_name \
--config config/dense_retrieval/tr-tripclick.yaml config/dense_retrieval/tr-out-of-domain.yaml \
--config-overwrites "trained_model: $trained_model_path"


#CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
#--run-name "beir_webis-touche2020-v2_"$trained_model_name \
#--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/beir/webis-touche2020/v2/config.yaml \
#--config-overwrites "trained_model: $trained_model_path"

#CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
#--run-name "beir_climate-fever_"$trained_model_name \
#--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/beir/climate-fever/config.yaml \
#--config-overwrites "trained_model: $trained_model_path"

#CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
#--run-name "beir_fiqa_"$trained_model_name \
#--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/beir/fiqa/config.yaml \
#--config-overwrites "trained_model: $trained_model_path"
#
#CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
#--run-name "beir_hotpotqa_"$trained_model_name \
#--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/beir/hotpotqa/config.yaml \
#--config-overwrites "trained_model: $trained_model_path"
#CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
#--run-name "natural-questions_"$trained_model_name \
#--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/natural-questions/config.yaml \
#--config-overwrites "trained_model: $trained_model_path"



CUDA_VISIBLE_DEVICES=$cuda_id python matchmaker/colberter_retrieval.py encode+index+search \
--run-name "beir_fiqa_"$trained_model_name \
--config config/dense_retrieval/tr-out-of-domain.yaml /newstorage5/shofstae/beir/fiqa/config.yaml \
--config-overwrites "trained_model: $trained_model_path"












#
#sanity check

CUDA_VISIBLE_DEVICES=5 python matchmaker/train.py --continue-folder /newstorage5/shofstae/msmarco-passage-experiments/2021-12-27_1432_colberter-v2_PL_dim128-32 \
--config-overwrites "validation_end: {},\
test:\n\
  sanity_trec2019_top1000:\n\
    tsv: /data01/hofstaetter/data/msmarco-passage/leaderboard/test2019.bm25_plain_top1000.tsv\n\
    qrels: /data01/hofstaetter/data/msmarco-passage/qrels/trec2019-qrels-pass.txt\n\
    binarization_point: 2\n\
    save_secondary_output: True\n\
  sanity_trec2020_bm25rerank:\n\
    tsv: /data01/hofstaetter/data/msmarco-passage/leaderboard/msmarco-passagetest2020-top1000.tsv\n\
    qrels: /data01/hofstaetter/data/msmarco-passage/qrels/2020-qrels-pass-final.txt\n\
    binarization_point: 2\n\
    save_secondary_output: True\n"



#
# get new test results from already trained models
#

CUDA_VISIBLE_DEVICES=1 python matchmaker/train.py --continue-folder /newstorage5/shofstae/msmarco-passage-experiments/2022-01-07_1936_colberter-v3_PL_dim128-32_weightedscore1_0 \
--config-overwrites "validation_end: {},\
test:\n\
  trec19+20-combined_tasb_top1k:\n\
    tsv: /home/shofstae/msmarco-data/passage-v1/trec19+20-combined-tasb-rerank-tuples.tsv\n\
    qrels: /home/shofstae/msmarco-data/passage-v1/trec19+20-combined-qrels.txt\n\
    binarization_point: 2\n\
    save_secondary_output: True\n\
  trec19+20-combined_pool-only:\n\
    tsv: /home/shofstae/msmarco-data/passage-v1/trec19+20-combined-poolonly-rerank-tuples.tsv\n\
    qrels: /home/shofstae/msmarco-data/passage-v1/trec19+20-combined-qrels.txt\n\
    binarization_point: 2\n\
    save_secondary_output: True"

CUDA_VISIBLE_DEVICES=1 python matchmaker/train.py --continue-folder /scratch/hofstaetter/experiments/msmarco-passage-v1/2022-01-06_1901_colberter-v3_PL_dim128-32_weightedscore0_5 \
--config-overwrites "validation_end: {},\
test:\n\
  trec19+20-combined_tasb_top1k:\n\
    tsv: /home/dlmain/shofstaetter/data/msmarco-passage/trec19+20-combined-tasb-rerank-tuples.tsv\n\
    qrels: /home/dlmain/shofstaetter/data/msmarco-passage/trec19+20-combined-qrels.txt\n\
    binarization_point: 2\n\
    save_secondary_output: True\n\
  trec19+20-combined_pool-only:\n\
    tsv: /home/dlmain/shofstaetter/data/msmarco-passage/trec19+20-combined-poolonly-rerank-tuples.tsv\n\
    qrels: /home/dlmain/shofstaetter/data/msmarco-passage/trec19+20-combined-qrels.txt\n\
    binarization_point: 2\n\
    save_secondary_output: True"


CUDA_VISIBLE_DEVICES=0 python matchmaker/train.py --continue-folder /newstorage5/shofstae/msmarco-passage-experiments/2022-01-06_1858_colberter-v3_PL_dim128-32_weightedscore0_0 \
--config-overwrites "validation_end: {},\
test:\n\
  top1000_6kdev_tasb256_rerank:\n\
    tsv: /data01/hofstaetter/data/msmarco-passage/tasb256_dev7k_tuples_top1k.tsv\n\
    qrels: /data01/hofstaetter/data/msmarco-passage/qrels/qrels.dev.tsv\n\
    binarization_point: 1\n\
    save_secondary_output: False\n\
  trec19+20-combined_tasb_top1k:\n\
    tsv: /data01/hofstaetter/data/msmarco-passage/trec19+20-combined-tasb-rerank-tuples.tsv\n\
    qrels: /data01/hofstaetter/data/msmarco-passage/trec19+20-combined-qrels.txt\n\
    binarization_point: 2\n\
    save_secondary_output: True\n\
  trec19+20-combined_pool-only:\n\
    tsv: /data01/hofstaetter/data/msmarco-passage/trec19+20-combined-poolonly-rerank-tuples.tsv\n\
    qrels: /data01/hofstaetter/data/msmarco-passage/trec19+20-combined-qrels.txt\n\
    binarization_point: 2\n\
    save_secondary_output: True"


#
# out-of-domain experiments -> bm25 baselines via pyserini & anserini-tools on troubadix
#

# selection criteria:
# - dataset need negative qrels
# - dataset needs to be public (except trec)
# - dataset needs > 50 queries

# selected datasets:

ours:

antique

trec-podcast (trec20+21)
tripclick (head-dctr)


from beir:

trec-robust04
trec-covid
dbpedia
nfcorpus
webis-touche2020/v2



python preprocessing/ir_datasets/get_for_dense_retrieval.py --out-folder /newstorage5/shofstae/ --dataset cranfield
python preprocessing/ir_datasets/get_for_dense_retrieval.py --out-folder /newstorage5/shofstae/ --dataset beir/nfcorpus 
python preprocessing/ir_datasets/get_for_dense_retrieval.py --out-folder /newstorage5/shofstae/ --dataset beir/webis-touche2020/v2 

# /newstorage5/shofstae/beir/trec-covid
# /newstorage5/shofstae/beir/climate-fever
# /newstorage5/shofstae/beir/dbpedia-entity
# /newstorage5/shofstae/beir/fiqa/
# /newstorage5/shofstae/beir/hotpotqa
# /newstorage5/shofstae/antique
# /newstorage5/shofstae/natural-questions


# scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/beir/trec-covid/all_qrels.txt trec-covid_qrels.txt
# scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/beir/climate-fever/all_qrels.txt climate-fever_qrels.txt
# scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/beir/dbpedia-entity/test_qrels.txt dbpedia-entity_qrels.txt
# scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/beir/fiqa/test_qrels.txt fiqa_qrels.txt
# scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/beir/hotpotqa/test_qrels.txt hotpotqa_qrels.txt
# scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/antique/test_qrels.txt antique_qrels.txt
# scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/natural-questions/dev_qrels.txt natural-questions_qrels.txt

#scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-16_1814_beir_fiqa_colberter-v3_PL_dim128-32_weightedscore0_1_unique-bow-stemmed/
#scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-16_1822_beir_hotpotqa_colberter-v3_PL_dim128-32_weightedscore0_1_unique-bow-stemmed/
#scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-16_1530_beir_climate-fever_colberter-v3_PL_dim128-32_weightedscore0_1_unique-bow-stemmed/

scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-18_1144_beir_trec-covid_tasb_dim768_hf/all_top1000-output.txt  tasb_dim768_hf/trec-covid_top1000-token-output.txt
scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-18_1146_beir_dbpedia-entity_tasb_dim768_hf/test_top1000-output.txt tasb_dim768_hf/dbpedia-entity_top1000-token-output.txt
scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-18_1213_beir_nfcorpus_tasb_dim768_hf/test_top1000-output.txt tasb_dim768_hf/nfcorpus_top1000-token-output.txt
scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-18_1214_antique_tasb_dim768_hf/test_top1000-output.txt tasb_dim768_hf/antique_top1000-token-output.txt
scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-18_1217_trec_podcast_tasb_dim768_hf/trec_podcast_20_21_combined-output.txt tasb_dim768_hf/trec_podcast_top1000-token-output.txt
scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-18_1301_tripclick_tasb_dim768_hf/top1k_head_dctr-output.txt tasb_dim768_hf/tripclick_top1000-token-output.txt

scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-21_1120_beir_trec-covid_colberter-exact_mini-v3_PL_dim128-32-1_weightedscore0_1_unique-bow-stemmed-stop0_75/all_top1000-token-output.txt  colberter_dim1_stopwords/trec-covid_top1000-token-output.txt
scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-21_1122_beir_dbpedia-entity_colberter-exact_mini-v3_PL_dim128-32-1_weightedscore0_1_unique-bow-stemmed-stop0_75/test_top1000-token-output.txt colberter_dim1_stopwords/dbpedia-entity_top1000-token-output.txt
scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-21_1146_beir_nfcorpus_colberter-exact_mini-v3_PL_dim128-32-1_weightedscore0_1_unique-bow-stemmed-stop0_75/test_top1000-token-output.txt colberter_dim1_stopwords/nfcorpus_top1000-token-output.txt
scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-21_1148_antique_colberter-exact_mini-v3_PL_dim128-32-1_weightedscore0_1_unique-bow-stemmed-stop0_75/test_top1000-token-output.txt colberter_dim1_stopwords/antique_top1000-token-output.txt
scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-21_1121_trec_podcast_colberter-exact_mini-v3_PL_dim128-32-1_weightedscore0_1_unique-bow-stemmed-stop0_75/trec_podcast_20_21_combined-token-output.txt colberter_dim1_stopwords/trec_podcast_top1000-token-output.txt
scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-01-21_1221_tripclick_colberter-exact_mini-v3_PL_dim128-32-1_weightedscore0_1_unique-bow-stemmed-stop0_75/top1k_head_dctr-token-output.txt colberter_dim1_stopwords/tripclick_top1000-token-output.txt

scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-02-08_1022_robust04_colberter-v3_PL_dim128-32_weightedscore0_1_unique-bow-stemmed_stop0_75/title_top1000-token-output.txt colberter_dim32_bowonly/robust04_top1000-token-output.txt
scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/colberter-out-of-domain-experiments/2022-02-08_0844_robust04_colberter-exact_mini-v3_PL_dim128-32-1_weightedscore0_1_unique-bow-stemmed-stop0_75/title_top1000-token-output.txt colberter_dim1_stopwords/robust04_top1000-token-output.txt
scp shofstae@troubadix.ifs.tuwien.ac.at:/mnt/nvme-local/hofstaetter/robust04-retrieval-experiments/2022-02-08_0846_robust04_tasb_dim768_hf/title_top1000-output.txt  tasb_dim768_hf/robust04_top1000-token-output.txt

scp shofstae@troubadix.ifs.tuwien.ac.at:/newstorage5/shofstae/out-of-domain-pyserini-runs/robust04_title.bm25-top1k.txt .

2022-01-18_1144_beir_trec-covid_tasb_dim768_hf
2022-01-18_1146_beir_dbpedia-entity_tasb_dim768_hf
2022-01-18_1213_beir_nfcorpus_tasb_dim768_hf
2022-01-18_1214_antique_tasb_dim768_hf
2022-01-18_1217_trec_podcast_tasb_dim768_hf
2022-01-18_1301_tripclick_tasb_dim768_hf


python analysis/evaluation/length_stats_collection.py --file /mnt/nvme-local/hofstaetter/msmarco-passage/collection.tsv > /newstorage5/shofstae/word_stats_msmarco.txt &
python analysis/evaluation/length_stats_collection.py --file /newstorage5/shofstae/beir/trec-covid/collection.tsv > /newstorage5/shofstae/word_stats_trec-covid.txt &
python analysis/evaluation/length_stats_collection.py --file /newstorage5/shofstae/beir/dbpedia-entity/collection.tsv > /newstorage5/shofstae/word_stats_dbpedia-entity.txt &
python analysis/evaluation/length_stats_collection.py --file /newstorage5/shofstae/antique/collection.tsv > /newstorage5/shofstae/word_stats_antique.txt &
python analysis/evaluation/length_stats_collection.py --file /newstorage5/shofstae/beir/nfcorpus/collection.tsv > /newstorage5/shofstae/word_stats_nfcorpus.txt &
python analysis/evaluation/length_stats_collection.py --file /newstorage4/hofstaetter/tripclick/benchmark/collection/collection.tsv > /newstorage5/shofstae/word_stats_tripclick.txt &
python analysis/evaluation/length_stats_collection.py --file /mnt/nvme-local/msertkan/spotify_podcast_track/data/preprocessed/collections/episode_title_and_segment_collection.tsv > /newstorage5/shofstae/word_stats_trec-podcast.txt &

python analysis/evaluation/length_stats_collection.py --file /newstorage5/shofstae/robust04/collection.tsv > /newstorage5/shofstae/word_stats_robust04.txt

#
# create pyserini input
#

python ~/anserini-tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path /newstorage5/shofstae/robust04/collection.tsv \
 --output-folder /newstorage5/shofstae/robust04/collection-jsonl

python ~/anserini-tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path /newstorage5/shofstae/beir/trec-covid/collection.tsv \
 --output-folder /newstorage5/shofstae/beir/trec-covid/collection-jsonl

python ~/anserini-tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path /newstorage5/shofstae/beir/climate-fever/collection.tsv \
 --output-folder /newstorage5/shofstae/beir/climate-fever/collection-jsonl

python ~/anserini-tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path /newstorage5/shofstae/beir/dbpedia-entity/collection.tsv \
 --output-folder /newstorage5/shofstae/beir/dbpedia-entity/collection-jsonl

python ~/anserini-tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path /newstorage5/shofstae/beir/fiqa/collection.tsv \
 --output-folder /newstorage5/shofstae/beir/fiqa/collection-jsonl

python ~/anserini-tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path /newstorage5/shofstae/beir/hotpotqa/collection.tsv \
 --output-folder /newstorage5/shofstae/beir/hotpotqa/collection-jsonl

python ~/anserini-tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path /newstorage5/shofstae/antique/collection.tsv \
 --output-folder /newstorage5/shofstae/antique/collection-jsonl

python ~/anserini-tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path /newstorage5/shofstae/natural-questions/collection.tsv \
 --output-folder /newstorage5/shofstae/natural-questions/collection-jsonl

python ~/anserini-tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path /newstorage5/shofstae/cranfield/collection.tsv \
 --output-folder /newstorage5/shofstae/cranfield/collection-jsonl
python ~/anserini-tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path /newstorage5/shofstae/beir/nfcorpus/collection.tsv \
 --output-folder /newstorage5/shofstae/beir/nfcorpus/collection-jsonl
python ~/anserini-tools/scripts/msmarco/convert_collection_to_jsonl.py \
 --collection-path /newstorage5/shofstae/beir/webis-touche2020/v2/collection.tsv \
 --output-folder /newstorage5/shofstae/beir/webis-touche2020/v2/collection-jsonl


#
# create pyserini index 
#

time python -m pyserini.index \
  --input /newstorage5/shofstae/robust04/collection-jsonl \
  --collection JsonCollection  --generator DefaultLuceneDocumentGenerator --index /newstorage5/shofstae/robust04/pyserini-index --threads 8 --storePositions

time python -m pyserini.index \
  --input /newstorage5/shofstae/beir/trec-covid/collection-jsonl \
  --collection JsonCollection  --generator DefaultLuceneDocumentGenerator --index /newstorage5/shofstae/beir/trec-covid/pyserini-index --threads 8 --storePositions

time python -m pyserini.index \
  --input /newstorage5/shofstae/beir/climate-fever/collection-jsonl \
  --collection JsonCollection  --generator DefaultLuceneDocumentGenerator --index /newstorage5/shofstae/beir/climate-fever/pyserini-index --threads 8 --storePositions

time python -m pyserini.index \
  --input /newstorage5/shofstae/beir/dbpedia-entity/collection-jsonl \
  --collection JsonCollection  --generator DefaultLuceneDocumentGenerator --index /newstorage5/shofstae/beir/dbpedia-entity/pyserini-index --threads 8 --storePositions

time python -m pyserini.index \
  --input /newstorage5/shofstae/beir/fiqa/collection-jsonl \
  --collection JsonCollection  --generator DefaultLuceneDocumentGenerator --index /newstorage5/shofstae/beir/fiqa/pyserini-index --threads 8 --storePositions

time python -m pyserini.index \
  --input /newstorage5/shofstae/beir/hotpotqa/collection-jsonl \
  --collection JsonCollection  --generator DefaultLuceneDocumentGenerator --index /newstorage5/shofstae/beir/hotpotqa/pyserini-index --threads 8 --storePositions

time python -m pyserini.index \
  --input /newstorage5/shofstae/antique/collection-jsonl \
  --collection JsonCollection  --generator DefaultLuceneDocumentGenerator --index /newstorage5/shofstae/antique/pyserini-index --threads 8 --storePositions

time python -m pyserini.index \
  --input /newstorage5/shofstae/natural-questions/collection-jsonl \
  --collection JsonCollection  --generator DefaultLuceneDocumentGenerator --index /newstorage5/shofstae/beir/natural-questions/pyserini-index --threads 8 --storePositions


time python -m pyserini.index \
  --input /newstorage5/shofstae/cranfield/collection-jsonl \
  --collection JsonCollection  --generator DefaultLuceneDocumentGenerator --index /newstorage5/shofstae/cranfield/pyserini-index --threads 8 --storePositions
time python -m pyserini.index \
  --input /newstorage5/shofstae/beir/nfcorpus/collection-jsonl \
  --collection JsonCollection  --generator DefaultLuceneDocumentGenerator --index /newstorage5/shofstae/beir/nfcorpus/pyserini-index --threads 8 --storePositions
time python -m pyserini.index \
  --input /newstorage5/shofstae/beir/webis-touche2020/v2/collection-jsonl \
  --collection JsonCollection  --generator DefaultLuceneDocumentGenerator --index /newstorage5/shofstae/beir/webis-touche2020/v2/pyserini-index --threads 8 --storePositions

#
# search for queries
#

time python -m pyserini.search --topics /newstorage5/shofstae/robust04/robust04_title_queries.tsv \
 --index /newstorage5/shofstae/robust04/pyserini-index \
 --output /newstorage5/shofstae/out-of-domain-pyserini-runs/robust04_title.bm25-top1k.txt \
 --bm25 --output-format msmarco --hits 1000

time python -m pyserini.search --topics /newstorage5/shofstae/beir/trec-covid/all_queries.tsv \
 --index /newstorage5/shofstae/beir/trec-covid/pyserini-index \
 --output /newstorage5/shofstae/out-of-domain-pyserini-runs/beir_trec-covid_all.bm25-top1k.txt \
 --bm25 --output-format msmarco --hits 1000

time python -m pyserini.search --topics /newstorage5/shofstae/beir/climate-fever/all_queries.tsv \
 --index /newstorage5/shofstae/beir/climate-fever/pyserini-index \
 --output /newstorage5/shofstae/out-of-domain-pyserini-runs/beir_climate-fever_all.bm25-top1k.txt \
 --bm25 --output-format msmarco --hits 1000

time python -m pyserini.search --topics /newstorage5/shofstae/beir/dbpedia-entity/test_queries.tsv \
 --index /newstorage5/shofstae/beir/dbpedia-entity/pyserini-index \
 --output /newstorage5/shofstae/out-of-domain-pyserini-runs/beir_dbpedia-entity_test.bm25-top1k.txt \
 --bm25 --output-format msmarco --hits 1000

time python -m pyserini.search --topics /newstorage5/shofstae/beir/fiqa/test_queries.tsv \
 --index /newstorage5/shofstae/beir/fiqa/pyserini-index \
 --output /newstorage5/shofstae/out-of-domain-pyserini-runs/beir_fiqa_test.bm25-top1k.txt \
 --bm25 --output-format msmarco --hits 1000

time python -m pyserini.search --topics /newstorage5/shofstae/beir/hotpotqa/test_queries.tsv \
 --index /newstorage5/shofstae/beir/hotpotqa/pyserini-index \
 --output /newstorage5/shofstae/out-of-domain-pyserini-runs/beir_hotpotqa_test.bm25-top1k.txt \
 --bm25 --output-format msmarco --hits 1000

time python -m pyserini.search --topics /newstorage5/shofstae/antique/test_queries.tsv \
 --index /newstorage5/shofstae/antique/pyserini-index \
 --output /newstorage5/shofstae/out-of-domain-pyserini-runs/antique_test.bm25-top1k.txt \
 --bm25 --output-format msmarco --hits 1000

time python -m pyserini.search --topics /newstorage5/shofstae/natural-questions/dev_queries.tsv \
 --index /newstorage5/shofstae/beir/natural-questions/pyserini-index \
 --output /newstorage5/shofstae/out-of-domain-pyserini-runs/natural-questions_dev.bm25-top1k.txt \
 --bm25 --output-format msmarco --hits 1000

time python -m pyserini.search --topics /newstorage5/shofstae/beir/nfcorpus/test_queries.tsv \
 --index /newstorage5/shofstae/beir/nfcorpus/pyserini-index \
 --output /newstorage5/shofstae/out-of-domain-pyserini-runs/beir_nfcorpus_test.bm25-top1k.txt \
 --bm25 --output-format msmarco --hits 1000
time python -m pyserini.search --topics /newstorage5/shofstae/beir/webis-touche2020/v2/all_queries.tsv \
 --index /newstorage5/shofstae/beir/webis-touche2020/v2/pyserini-index \
 --output /newstorage5/shofstae/out-of-domain-pyserini-runs/beir_webis-touche2020-v2_all.bm25-top1k.txt \
 --bm25 --output-format msmarco --hits 1000
time python -m pyserini.search --topics /newstorage5/shofstae/cranfield/all_queries.tsv \
 --index /newstorage5/shofstae/cranfield/pyserini-index \
 --output /newstorage5/shofstae/out-of-domain-pyserini-runs/cranfield_all.bm25-top1k.txt \
 --bm25 --output-format msmarco --hits 1000

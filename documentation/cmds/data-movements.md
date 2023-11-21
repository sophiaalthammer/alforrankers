# copy msmarco to idefix

target=shofstae@idefix.isis.tuwien.ac.at:~/msmarco-data/passage-v1/.

scp /mnt/nvme-local/hofstaetter/msmarco-passage/queries.train.tsv \
/mnt/nvme-local/hofstaetter/msmarco-passage/collection.tsv \
/data01/hofstaetter/msmarco-passage-experiments/3bert_ensemble_hardnegs/test_tasb_train-ensemble-output.txt \
/data01/hofstaetter/data/msmarco-passage/validation/T2_smart_earlystopping3kq_top100.tsv \
/data01/hofstaetter/data/msmarco-passage/qrels/qrels.dev.tsv \
/data01/hofstaetter/data/msmarco-passage/tasb256_dev7k_tuples_top1k.tsv \
/data01/hofstaetter/data/msmarco-passage/leaderboard/test2019.bm25_plain_top1000.tsv \
/data01/hofstaetter/data/msmarco-passage/qrels/trec2019-qrels-pass.txt \
/data01/hofstaetter/data/msmarco-passage/leaderboard/msmarco-passagetest2020-top1000.tsv \
/data01/hofstaetter/data/msmarco-passage/qrels/2020-qrels-pass-final.txt $target


target=shofstae@idefix.isis.tuwien.ac.at:~/msmarco-data/passage-v1/.

scp /data01/hofstaetter/data/msmarco-passage/queries/queries.dev.judged.subset.tsv \
/data01/hofstaetter/data/msmarco-passage/queries/msmarco-test2019-queries.tsv \
/data01/hofstaetter/data/msmarco-passage/leaderboard/msmarco-test2020-queries.tsv $target


import os

path = '/mnt/c/Users/salthamm/Documents/phd/data/msmarco-passage-v2/retrieval_experiments/2022-08-05_0631_uni-colberter-128-1-msmarco'

run_id = 'tuvienna-pas-unicol'

with open(os.path.join(path, 'trec2022_top1000-token-output.txt'),'r') as in_file, \
        open(os.path.join(path, 'trec2022_top100_run2.txt'), 'w') as out_file:
    lines = in_file.readlines()

    for line in lines:
        splitted = line.split('\t')
        query_id = splitted[0]
        doc_id = splitted[1]
        rank = splitted[2]
        score = splitted[3].rstrip('\n')

        # only submit top 100
        if int(rank) < 101:
            out_file.write(query_id + '  ' + 'Q0' + '  ' + doc_id + '  ' + rank + '  ' + score + '  ' + run_id + '\n')





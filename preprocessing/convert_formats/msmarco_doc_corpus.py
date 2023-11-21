import os
import gzip
import json

path = '/newstorage4/salthamm/msmarco-doc-v2/msmarco_v2_doc'

files = os.listdir(path)
with open(os.path.join(path, 'collection_all.tsv'), 'w') as out_file:
    for file in files:
        print('start with {}'.format(file))
        with gzip.open(os.path.join(path, file), 'r') as in_file:
        #with open(os.path.join(path, file), 'r') as in_file:
            json_list = list(in_file)

            for json_str in json_list:
                result = json.loads(json_str)
                doc_id = result.get('docid')
                title = result.get('title')
                title = title.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
                headings = result.get('headings')
                headings = headings.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
                body = result.get('body')
                body = body.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()

                text = title + body
                text = ' '.join(text.split(' ')[:200])

                out_file.write(doc_id + '\t' + text + '\n')



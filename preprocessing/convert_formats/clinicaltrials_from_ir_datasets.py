import ir_datasets
import csv
import os

output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/clinical-trials'  #"/newstorage5/salthamm/clinical_trials/data" #'/mnt/c/Users/salthamm/Documents/phd/data/clinical-trials'

# collection
dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")
with open(os.path.join(output_dir, 'collection_all_content.tsv'), 'w',encoding="utf8") as out_file:
    #, \
    #open(os.path.join(output_dir,"collection_title_summary.tsv"),"w",encoding="utf8") as out_file2:
#with open(os.path.join(output_dir, 'docid_eligibility.tsv'), 'w', encoding="utf8") as outfile3:
    for doc in dataset.docs_iter():
        title_text = doc.title
        title_text = title_text.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
        condition_text = str(doc.condition)
        condition_text = condition_text.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
        elig_text = str(doc.eligibility)
        elig_text = elig_text.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
        summary_text = str(doc.summary)
        summary_text = summary_text.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
        detailed_desc = str(doc.detailed_description)
        detailed_desc = detailed_desc.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()

        out_file.write(doc.doc_id + "\t" + title_text + ' ' + condition_text +' ' + summary_text + ' '
                       + detailed_desc + ' ' + elig_text + "\n")
        #out_file2.write(doc.doc_id + "\t" + title_text + ' ' + summary_text + "\n")
        #outfile3.write(doc.doc_id + "\t" + doc.eligibility + "\n")

#
# # queries 2021
# dataset_queries = ir_datasets.load("clinicaltrials/2021/trec-ct-2022")   #clinicaltrials/2021/trec-ct-2021
#
# with open(os.path.join(output_dir, 'queries_2022.tsv'), 'wt') as out_file:
#    for query in dataset_queries.queries_iter():
#        out_file.write(query.query_id + "\t" + query.text.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip() +"\n")
#
# # giorgios inclusion exclusion texts
# elig_dict = {}
# for doc in dataset.docs_iter():
#     text = str(doc.eligibility)
#     new_text = text.lstrip().replace(r'\s++',' ').replace(r'\t++',' ')
#     new_text2 = new_text.replace("\t", " ").replace("\n", " ").replace("\r", " ").replace(";", "").strip()
#     #new_text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
#     #new_text2 = ''.join(new_text.splitlines())
#     splitted = new_text2.split()
#     splitted2 = ' '.join(splitted)
#     elig_dict.update({doc.doc_id:splitted2})
#
# with open(os.path.join(output_dir, 'data/trec21_incl_excl.csv'), 'r') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter='\t')
#     lines = []
#     for line in csv_reader:
#         lines.append(line)
#         assert len(line) == 6
#
# print(lines[0])
# elig = []
# lines[0][5] = 'eligibility'
# elig.append(lines[0])
#
# for line in lines:
#     if line[1] != 'docno':
#         eligibity_text = str(elig_dict.get(line[1]))
#         line[5] = eligibity_text
#         elig.append(line)
#         assert len(line) == 6
#
# with open(os.path.join(output_dir, 'data/trec21_incl_excl_elig.csv'), 'w') as csv_file:
#     csv_writer = csv.writer(csv_file, delimiter='\t')
#     for line in elig:
#         csv_writer.writerow(line)
#
#
# # now write tsv with:
# # id \t text \n
# with open(os.path.join(output_dir, 'inclusion_crit.tsv'), 'wt') as out_file:
#     for line in lines:
#         out_file.write(line[1] + "\t" + line[3].replace("\t"," ").replace("\n"," ").replace("\r"," ").strip() +"\n")
#
# with open(os.path.join(output_dir, 'exclusion_crit.tsv'), 'wt') as out_file:
#     for line in lines:
#         out_file.write(line[1] + "\t" + line[4].replace("\t"," ").replace("\n"," ").replace("\r"," ").strip() +"\n")
#
# with open(os.path.join(output_dir, 'summary_crit.tsv'), 'wt') as out_file:
#     for line in lines:
#         out_file.write(line[1] + "\t" + line[2].replace("\t"," ").replace("\n"," ").replace("\r"," ").strip() +"\n")
#
#
# # connect to one file
# with open(os.path.join(output_dir, 'queries_top1000-output-tasb-inclusion.txt'), 'r') as f:
#     lines = f.readlines()
#     data = {}
#     for line in lines:
#         splitted = line.strip('\n').split('\t')
#         if data.get(splitted[0]):
#             data.get(splitted[0]).update({splitted[1]: [splitted[3]]})
#         else:
#             data.update({splitted[0]:{}})
#             data.get(splitted[0]).update({splitted[1]: [splitted[3]]})
#
# print('done with first')
# with open(os.path.join(output_dir, 'queries_top1000-output-tasb-exclusion.txt'), 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         splitted = line.strip('\n').split('\t')
#         in_score = data.get(splitted[0]).get(splitted[1])
#         in_score.append(splitted[3])
#         data.get(splitted[0]).update({splitted[1]: in_score})
#
# print('done with second')
# with open(os.path.join(output_dir, 'queries_top1000-output-tasb-summary.txt'), 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         splitted = line.strip('\n').split('\t')
#         in_score = data.get(splitted[0]).get(splitted[1])
#         in_score.append(splitted[3])
#         data.get(splitted[0]).update({splitted[1]: in_score})
#
# print('done with third')
# with open(os.path.join(output_dir,'concat_3_scores_tasb.csv'), 'w') as f:
#     writer = csv.writer(f)
#     for query_id, value in data.items():
#         for doc_id, scores in value.items():
#             writer.writerow([query_id, doc_id, scores[0], scores[1], scores[2]])

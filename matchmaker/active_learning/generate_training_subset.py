import os
import random

def load_file(path):
    collection = {}
    with open(path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            splitted = line.split('\t')
            collection.update({splitted[0]: splitted[1].rstrip('\n')})
    return collection


def load_qrels(path):
    with open(path,'r') as f:
        qids_to_relevant_passageids = {}
        for l in f:
            try:
                l = l.strip().split()
                qid = l[0]
                if float(l[3]) > 0.0001:
                    if qid not in qids_to_relevant_passageids:
                        qids_to_relevant_passageids[qid] = {}
                    qids_to_relevant_passageids[qid][l[2]] = float(l[3])
            except:
                raise IOError('\"%s\" is not valid format' % l)
        return qids_to_relevant_passageids


def generate_train_subset(collection_path, train_queries_path, qrels_path, bm25_file, run_folder, traindata_size, neg, random_seed):
    random.seed(random_seed)
    # load the collection
    collection = load_file(collection_path)
    print('loaded the collection')

    # open the qrels
    qrels = load_qrels(qrels_path)
    # loads only relevant docs

    # load the train queries
    queries_train = load_file(train_queries_path)
    # maybe first select the train queries which are in the qrels, yes thats a good idea!
    queries_train_qrels = list(set(queries_train.keys()).intersection(set(qrels.keys())))

    # sample randomly traindata_size from train queries
    random_queries = random.sample(queries_train_qrels, traindata_size)
    print('loaded queries and sampled {} randomly'.format(traindata_size))

    # load neg from bm25 in case i need them
    if neg == 'bm25':
        bm25_neg = {}
        with open(bm25_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                query_id = line[0]
                doc_id = line[2]
                if query_id in random_queries:
                    if bm25_neg.get(query_id):
                        doc_list = bm25_neg.get(query_id)
                        # only add the negative documents
                        if doc_id not in qrels.get(query_id).keys():
                            doc_list.append(doc_id)
                            bm25_neg.update({query_id: doc_list})
                    else:
                        bm25_neg.update({query_id: [doc_id]})
                else:
                    print('error this query id {} is not in the plain_bm25 queries'.format(query_id))

    print('starting to generate the new training files')
    with open(os.path.join(run_folder, 'train.subset.{}.{}neg.tsv'.format(traindata_size, neg)), 'w') as out_file, \
            open(os.path.join(run_folder, 'train.subset.{}.{}neg.ids.tsv'.format(traindata_size, neg)),
                 'w') as out_file2:
        for query_id in random_queries:
            query_text = queries_train.get(query_id)

            try:
                # get the positive sample and the positive text
                pos_id = list(qrels.get(query_id).keys())[0]
                pos_text = collection.get(pos_id)

                # get the negative sample: either from bm25 or random from the collection which is not the positive one and the text
                if neg == 'bm25':
                    if bm25_neg.get(query_id):
                        doc_bm25 = bm25_neg.get(query_id)
                        neg_id = random.sample(doc_bm25, 1)[0]
                    else:
                        neg_ids = random.sample(list(collection.keys()), 2)

                        if neg_ids[0] == pos_id:
                            neg_id = neg_ids[1]
                        else:
                            neg_id = neg_ids[0]
                    neg_text = collection.get(neg_id)
                else:
                    neg_ids = random.sample(list(collection.keys()), 2)

                    if neg_ids[0] == pos_id:
                        neg_id = neg_ids[1]
                    else:
                        neg_id = neg_ids[0]
                    neg_text = collection.get(neg_id)

                # write out to the outfile with texts
                # write out the train query ids as well
                out_file.write(query_text + '\t' + pos_text + '\t' + neg_text + '\n')
                out_file2.write(query_id + '\n')
            except:
                print('error for query id {}'.format(query_id))

    return os.path.join(run_folder, 'train.subset.{}.{}neg.tsv'.format(traindata_size, neg))

def generate_train_subset_from_train(train_tsv_path, run_folder, traindata_size, random_seed, remain=False):

    random.seed(random_seed)
    with open(train_tsv_path, 'r') as f, open(os.path.join(run_folder, 'triples.train.subset.{}.tsv'.format(traindata_size)), 'w') as out_file:
        lines = f.readlines()
        random_lines = random.sample(lines, traindata_size)

        for line in random_lines:
            out_file.write(line)

        if remain:
            remain_lines = [line for line in lines if line not in random_lines]
            with open(os.path.join(run_folder, 'triples.train.remain.subset.tsv'), 'r') as remain_file:
                for line in remain_lines:
                    remain_file.write(line)

    return os.path.join(run_folder, 'triples.train.subset.{}.tsv'.format(traindata_size))

def generate_train_subset_from_train_firstk(train_tsv_path, run_folder, traindata_size, remain=False):

    with open(train_tsv_path, 'r') as f, open(os.path.join(run_folder, 'triples.train.subset.{}.tsv'.format(traindata_size)), 'w') as out_file:
        lines = f.readlines()
        lines = lines[:traindata_size]

        print('length of lines is now {}'.format(len(lines)))

        for line in lines:
            out_file.write(line)

        if remain:
            remain_lines = [line for line in lines if line not in lines]
            with open(os.path.join(run_folder, 'triples.train.remain.subset.tsv'), 'r') as remain_file:
                for line in remain_lines:
                    remain_file.write(line)

    return os.path.join(run_folder, 'triples.train.subset.{}.tsv'.format(traindata_size))

def generate_train_subset_incrementally(train_tsv_path, run_folder, traindata_size, random_seed, exp_path,
                                        previous_experiment_name, warmstart_model=False):
    previous_run_folder = None
    previous_train_file = None
    previous_traindata_size = None
    previous_best_model = None

    dirs = os.listdir(exp_path)
    for dir in dirs:
        if os.path.isdir(os.path.join(exp_path, dir)) and dir.endswith(previous_experiment_name):
            list_files = os.listdir(os.path.join(exp_path, dir))
            for file in list_files:
                if file.startswith('triples.train.subset'):
                    previous_traindata_size = int(file[21:-4])
                    previous_run_folder = os.path.join(exp_path, dir)
                    previous_train_file = os.path.join(previous_run_folder, file)
                if file.startswith('best-model.pytorch-state-dict'):
                    previous_run_folder = os.path.join(exp_path, dir)
                    previous_best_model = os.path.join(previous_run_folder, file)
            if previous_train_file is None:
                raise Exception('couldnt find the training triple in the previous run folder')
    if previous_train_file is None:
        raise Exception('couldnt find the previous run folder in the expirement base path')

    # create training datasize from train with the size
    if previous_run_folder is not None and previous_traindata_size is not None and previous_train_file is not None:

        random.seed(random_seed)
        with open(train_tsv_path, 'r') as f, open(
                os.path.join(run_folder, 'triples.train.subset.{}.tsv'.format(traindata_size)), 'w') as out_file, \
                open(previous_train_file, 'r') as previous_train:
            # first the previous subset
            lines_train = previous_train.readlines()
            for line in lines_train:
                out_file.write(line)

            # then add new training samples from the ones which are not added yet
            lines = f.readlines()
            lines = [x for x in lines if x not in lines_train]

            if traindata_size > previous_traindata_size:
                random_lines = random.sample(lines, int(traindata_size - previous_traindata_size))
                for line in random_lines:
                    out_file.write(line)

            else:
                raise Exception('next training must have bigger training datasize than the previous one')

    if previous_best_model is not None and warmstart_model:
        return os.path.join(run_folder, 'triples.train.subset.{}.tsv'.format(topic_no)), previous_best_model
    else:
        return os.path.join(run_folder, 'triples.train.subset.{}.tsv'.format(traindata_size))


def generate_subset_control_topic_no(train_tsv_path, run_folder, topic_no, random_seed, triplet_no_per_topic):
    random.seed(random_seed)
    with open(train_tsv_path, 'r') as f, open(os.path.join(run_folder, 'triples.train.subset.{}.tsv'.format(topic_no)),
                                              'w') as out_file:
        lines = f.readlines()

        topics = []
        samples_per_topic = {}
        for line in lines:
            query_text, pos_text, neg_text = line.split('\t')
            neg_text = neg_text.replace('\n', '')

            topics.append(query_text)

            if query_text not in samples_per_topic.keys():
                samples_per_topic.update({query_text: []})
            samples_per_topic[query_text].append((pos_text, neg_text))

        topics = list(set(topics))

        random_lines = random.sample(topics, topic_no)

        samples_train = []
        for topic in random_lines:
            samples = samples_per_topic.get(topic)
            i = 0
            for sample in samples:
                if i < triplet_no_per_topic:
                    samples_train.append(topic + '\t' + sample[0] + '\t' + sample[1] + '\n')
                    #out_file.write(topic + '\t' + sample[0] + '\t' + sample[1] + '\n')
                    i += 1
        random.shuffle(samples_train)
        for sample_train in samples_train:
            out_file.write(sample_train)
    return os.path.join(run_folder, 'triples.train.subset.{}.tsv'.format(topic_no))


def generate_subset_control_incrementally(train_tsv_path, run_folder, topic_no, random_seed, triplet_no_per_topic,exp_path,
                                        previous_experiment_name, warmstart_model=False):
    previous_run_folder = None
    previous_train_file = None
    previous_traindata_size = None
    previous_best_model = None

    dirs = os.listdir(exp_path)
    for dir in dirs:
        if os.path.isdir(os.path.join(exp_path, dir)) and dir.endswith(previous_experiment_name):
            list_files = os.listdir(os.path.join(exp_path, dir))
            for file in list_files:
                if file.startswith('triples.train.subset'):
                    previous_traindata_size = int(file[21:-4])
                    previous_run_folder = os.path.join(exp_path, dir)
                    previous_train_file = os.path.join(previous_run_folder, file)
                if file.startswith('best-model.pytorch-state-dict'):
                    previous_best_model = os.path.join(previous_run_folder, file)
            if previous_train_file is None:
                raise Exception('couldnt find the training triple in the previous run folder')
    if previous_train_file is None:
        raise Exception('couldnt find the previous run folder in the expirement base path')

    # create training datasize from train with the size
    if previous_run_folder is not None and previous_traindata_size is not None and previous_train_file is not None:

        random.seed(random_seed)

        random.seed(random_seed)
        with open(train_tsv_path, 'r') as f, open(
                os.path.join(run_folder, 'triples.train.subset.{}.tsv'.format(topic_no)),
                'w') as out_file, open(previous_train_file, 'r') as previous_train:

            lines_train = previous_train.readlines()
            previous_topics = []
            previous_samples_no = 0
            for line in lines_train:
                previous_topics.append(line.split('\t')[0])
                previous_samples_no += 1
                out_file.write(line)

            previous_triplet_no_per_topic = int(previous_samples_no/previous_traindata_size)
            previous_topics = list(set(previous_topics))

            # these lines are the ones which are not in the training set yet, but they could be for old topics!
            lines = f.readlines()
            lines = [x for x in lines if x not in lines_train]

            topics = []
            samples_per_topic = {}
            for line in lines:
                query_text, pos_text, neg_text = line.split('\t')
                neg_text = neg_text.replace('\n', '')

                topics.append(query_text)

                if query_text not in samples_per_topic.keys():
                    samples_per_topic.update({query_text: []})
                samples_per_topic[query_text].append((pos_text, neg_text))

            # so that only new topics are chosen
            topics_wo_previous = list(set(topics).difference(set(previous_topics)))

            if topic_no > previous_traindata_size:
                random_topics = random.sample(topics_wo_previous, int(topic_no - previous_traindata_size))
                #for line in random_lines:
                #    out_file.write(line)
                samples_train = []
                for topic in random_topics:
                    samples = samples_per_topic.get(topic)
                    i = 0
                    for sample in samples:
                        if i < triplet_no_per_topic:
                            samples_train.append(topic + '\t' + sample[0] + '\t' + sample[1] + '\n')
                            #out_file.write(topic + '\t' + sample[0] + '\t' + sample[1] + '\n')
                            i += 1

                random.shuffle(samples_train)
                for sample_train in samples_train:
                    out_file.write(sample_train)
                print('added new topics for the training')
            elif triplet_no_per_topic > previous_triplet_no_per_topic:
                samples_train = []
                for topic in previous_topics:
                    if samples_per_topic.get(topic) is not None:
                        samples = samples_per_topic.get(topic)
                        i = 0
                        for sample in samples:
                            if i < int(triplet_no_per_topic-previous_triplet_no_per_topic):
                                samples_train.append(topic + '\t' + sample[0] + '\t' + sample[1] + '\n')
                                # out_file.write(topic + '\t' + sample[0] + '\t' + sample[1] + '\n')
                                i += 1

                random.shuffle(samples_train)
                for sample_train in samples_train:
                    out_file.write(sample_train)
                print('added training files for no new topics, but new samples per topic')
            else:
                raise Exception('next training must have bigger training datasize than the previous one')

    if previous_best_model is not None and warmstart_model:
        return os.path.join(run_folder, 'triples.train.subset.{}.tsv'.format(topic_no)), previous_best_model
    else:
        return os.path.join(run_folder, 'triples.train.subset.{}.tsv'.format(topic_no))


if __name__ == "__main__":
    random_seed = 876437631
    path = "/mnt/c/Users/salthamm/Documents/phd/data/msmarco-passage/data" #"/scratch/salthammer/data/msmarco" #"/newstorage5/salthamm/msmarco/data" #"/mnt/c/Users/salthamm/Documents/phd/data/msmarco-passage/data"

    topic_no = 11
    triplet_no_per_topic = 10
    traindata_size = 200
    neg =  'random' #'bm25'  # or 'random'

    qrels_path = os.path.join(path, 'qrels.train.tsv')
    collection_path = os.path.join(path, 'collection.tsv')
    train_queries_path = os.path.join(path, "data_efficiency/queries.train.wotest.tsv")
    bm25_file = os.path.join(path, 'plain_bm25_train_top100.txt')
    run_folder = path #"/home/dlmain/salthammer/experiments/msmarco/generate_train_subset" #path

    #generate_train_subset(collection_path, train_queries_path, qrels_path, bm25_file, run_folder, traindata_size, neg, random_seed)

    # new function: generate subset from given training dataset

    train_tsv_path = os.path.join(path, 'train/train.subset.1000.randomneg.tsv')
    #generate_train_subset_from_train(train_tsv_path, run_folder, traindata_size)

    # increase incrementally
    exp_path = os.path.join(path, "data_efficiency")  # config['expirement_base_path']
    previous_experiment_name = 'train_subset_5'  # config['previous_run_name']

    #generate_train_subset_incrementally(train_tsv_path, run_folder, traindata_size, 12782, exp_path, previous_experiment_name)


    # control the topic consistency as well!
    #generate_subset_control_topic_no(train_tsv_path, run_folder, topic_no, 2783, triplet_no_per_topic)

    #generate_subset_control_incrementally(train_tsv_path, run_folder, topic_no, random_seed, triplet_no_per_topic,
    #                                      exp_path,
    #                                      previous_experiment_name)



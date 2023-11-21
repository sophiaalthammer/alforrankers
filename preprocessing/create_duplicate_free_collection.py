import numpy as np
import argparse
import os
import sys
sys.path.append(os.getcwd())

from matchmaker.dataloaders.ir_single_sequence_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.common import Tqdm

from matchmaker.utils import *

from matchmaker.dataloaders.bling_fire_tokenizer import BlingFireTokenizer



import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

# from: https://github.com/FreeDiscovery/FreeDiscovery/blob/master/freediscovery/near_duplicates/simhash.py
# Authors: Roman Yurchak
#
# License: BSD 3 clause

class SimhashNearDuplicates(BaseEstimator):
    """Near duplicates detection using the simhash algorithm.

    A classical near-duplicates detection involves comparing all pairs of
    samples in the collection. For a collection of size ``N``, this is
    typically an ``O(N^2)`` operation. Simhash algorithm allows to
    retrieve near duplicates with a significantly better computational
    scaling.

    .. Note:: this estimator requires
              the `simhash-py <https://github.com/seomoz/simhash-py>`_Python package
              to be installed.

    Parameters
    ----------
    hash_func : str or function, default='murmurhash3_int_u32'
        the hashing function used to hash documents.
        Possibles values are "murmurhash3_int_u32" or a custom function.
    hash_func_nbytes : int, default=64
        expected size of the hash produced by hash_func

    References
    ----------
    .. [Charikar2002]  `Charikar, M. S. (2002, May).
                        Similarity estimation techniques from rounding
                        algorithms.
                        In Proceedings of the thiry-fourth annual ACM symposium
                        on Theory of computing (pp. 380-388). ACM.`
    """
    def __init__(self, hash_func='murmurhash3_int_u32', hash_func_nbytes=32):
        self._fit_X = None
        self._fit_shash_dict = {}
        if isinstance(hash_func, str):
            if hash_func == 'murmurhash3_int_u32':
                from sklearn.utils.murmurhash import murmurhash3_int_u32
                hash_func = murmurhash3_int_u32
                hash_func_nbytes = 32
            else:
                raise ValueError
        elif not hasattr(hash_func, '__call__'):
            raise ValueError

        self.hash_func = hash_func
        if hash_func_nbytes not in [32, 64]:
            raise ValueError('Hashing function other than 64bit '
                             'or 32bit are not supported!')

        self.hash_func_nbytes = hash_func_nbytes

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : {array, sparse matrix}, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        self : object
            Returns self.
        """
        from simhash import compute
        self._fit_X = X = check_array(X, accept_sparse='csr')

        n_features = X.shape[1]

        def _scale_hash_32_64bit(indices):
            return indices*((2**64-1)//2**32-1)

        hash_func = self.hash_func

        hashing_table = np.array(
                [hash_func(el, 0) for el in range(n_features)],
                dtype='uint64')

        shash = []
        for idx in range(X.shape[0]):
            # get hashes of indices
            mhash = hashing_table[X[idx].indices]
            if self.hash_func_nbytes == 32:
                mhash = _scale_hash_32_64bit(mhash)
            shash.append(compute(mhash))
        _fit_shash = np.asarray(shash, dtype='uint64')
        self._fit_shash = _fit_shash
        self._fit_shash_dict = {val: key
                                for key, val in enumerate(self._fit_shash)}

    def get_index_by_hash(self, shash):
        """ Get document index by hash

        Parameters
        ----------
        shash: uint64
           a simhash value

        Returns
        -------
        index: int
           a document index
        """
        return self._fit_shash_dict[shash]

    def query(self, distance=2, blocks='auto'):
        """ Find all the nearests neighbours for the dataset

        Parameters
        ----------
        distance : int, default=2
            Maximum number of differnet bits in the simhash
        blocks : int or 'auto', default='auto'
                number of blocks into which the simhash is split
                when searching for duplicates,
                see  https://github.com/seomoz/simhash-py

        Returns
        -------
        simhash : array
            the simhash value for all documents in the collection
        cluster_id : array
            the exact duplicates (documents with the same simhash)
            are grouped by in cluster_id
        dup_pairs : list
            list of tuples for the near-duplicates
        """
        from simhash import find_all

        if distance >= 64:
            raise ValueError(('Wrong input parameter for distance = {} '
                              'Must be less than 64!')
                             .format(distance))

        _, cluster_id = np.unique(self._fit_shash, return_inverse=True)

        if blocks == 'auto':
            blocks = min(distance*2, 64)
        matches = find_all(self._fit_shash, blocks, distance)
        matches = np.array(matches, dtype='uint64')
        return self._fit_shash, cluster_id, matches

#### ---------------

#
# config
#
parser = argparse.ArgumentParser()


parser.add_argument('--dataset-file', action='store', dest='dataset_file',
                    help='dataset file: for triple loader', required=True)
parser.add_argument('--vocab-file', action='store', dest='vocab_file',
                    help='vocab directory path', required=True)

parser.add_argument('--unique-out', action='store', dest='u_out',
                    help='', required=True)

parser.add_argument('--duplicate-out', action='store', dest='d_out',
                    help='', required=True)

args = parser.parse_args()

from sklearn.feature_extraction.text import HashingVectorizer


fe = HashingVectorizer(ngram_range=(4, 4), analyzer='word')

sequences = []
sequence_ids = []

with open(args.dataset_file,encoding="utf8") as file:
    for i,line in Tqdm.tqdm(enumerate(file)):

        line=line.split("\t")

        sequence_ids.append(line[0])
        sequences.append(line[1].strip())

        #if i == 2000:
        #    break

sequence_data = fe.fit_transform(sequences)

#loader = IrSingleSequenceDatasetReader(lazy=True,tokenizer=BlingFireTokenizer(),max_seq_length=200,min_seq_length=200)
#instances = loader.read(args.dataset_file)
#_iterator = BucketIterator(batch_size=64,
#                           sorting_keys=[("seq_tokens", "num_tokens")])
#_iterator.index_with(Vocabulary.from_files(args.vocab_file))
#
#sequence_ids = []
#sequence_data = []
#
#for i,inst in Tqdm.tqdm(enumerate(_iterator(instances, num_epochs=1))):
#    sequence_ids.extend(inst["seq_id"])
#    sequence_data.append(np.int32(inst["seq_tokens"]["tokens"].numpy()))
#    if i == 20:
#        break
#
#from scipy.sparse import csr_matrix
#
#sequence_data = csr_matrix(numpy.concatenate(sequence_data,axis=0))

print(sequence_data.shape)

simhash = SimhashNearDuplicates()

simhash.fit(sequence_data)

hash_values,exact_duplicate,near_duplicates = simhash.query(distance=5)

print(exact_duplicate.shape[0] - np.unique(exact_duplicate).shape[0])

#u, i = np.unique(exact_duplicate, return_inverse=True)
#print(u[np.bincount(i) > 1])
#print(exact_duplicate[u[np.bincount(i) > 1]])
#print(np.argwhere(np.bincount(i) > 1))




from collections import Counter
duplicated_cluster_ids = [item for item, count in Counter(exact_duplicate).items() if count > 1]
exact_duplicates = []
for dup in duplicated_cluster_ids:
    exact_duplicates.extend(np.argwhere(exact_duplicate == dup).flatten().tolist())
print(len(exact_duplicates))

near_duplicate_ids = []
for a,b in near_duplicates:
    near_duplicate_ids.append(simhash.get_index_by_hash(a))
    near_duplicate_ids.append(simhash.get_index_by_hash(b))

print(len(near_duplicate_ids))

all_duplicates = exact_duplicates + near_duplicate_ids
all_duplicate_set = set(all_duplicates)

print("duplicate-count:",len(all_duplicates))
print("duplicate-set-count:",len(all_duplicate_set))

with open(args.d_out,"w",encoding="utf8") as out_file_d,\
     open(args.u_out,"w",encoding="utf8") as out_file_u:
    for i,s in enumerate(sequence_ids):
        if i in all_duplicate_set:
            out_file_d.write(s+"\t"+sequences[i]+"\n")
        else:
            out_file_u.write(s+"\t"+sequences[i]+"\n")

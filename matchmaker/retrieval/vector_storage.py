import glob
import os
from typing import Any, Dict, Iterator, List
import numpy

from matchmaker.utils.utils import saveCompressed

class VectorStorage():
    '''
    Dynamic memory mapped vector + info storage
    '''

    def __init__(self, name, run_folder, config, dimensions):
        super(VectorStorage, self).__init__()
        self.name = name
        self.vector_dimensions = dimensions
        self.token_dtype = config["storage_"+self.name+"_dtype"]
        #self.use_fp16 = config["storage_"+self.name+"_dtype"] == "float16"
        self.run_folder = run_folder

        self.doc_infos = {}

        self.token_base_size = config["storage_"+self.name+"_block_size"]
        self.token_base_number = -1
        self.token_base = None 

        self.current_ids = None
        self.id_mapping = []
        self.storage = []
        self.storage_filled_to_index = []

    def add(self, vectors:numpy.ndarray):
        '''
        Add either a 1d or 2d vector to the storage
        '''
        if len(vectors.shape) == 1:
            vec_count = 1
        else:
            vec_count = vectors.shape[0]

        #
        # 2 options: 1) first add, 2) current store is full
        #
        if self.token_base_number < 0 or self.storage_filled_to_index[-1] + vec_count > self.token_base_size:
            
            self.storage_filled_to_index.append(0)

            self.token_base_number+=1
            self.token_base = numpy.memmap(os.path.join(self.run_folder,self.name+"_storage_"+str(self.token_base_number)+".npy"), dtype=numpy.dtype(self.token_dtype),
                                           mode="w+", shape=(self.token_base_size,self.vector_dimensions))
            self.current_ids = numpy.ndarray(shape=(self.token_base_size), dtype = 'int64') 

            self.storage.append(self.token_base)
            self.id_mapping.append(self.current_ids)

        start_index = self.storage_filled_to_index[-1]
        self.storage_filled_to_index[-1] = self.storage_filled_to_index[-1] + vec_count
        self.token_base[start_index:self.storage_filled_to_index[-1]] = vectors

        insert_idx = len(self.doc_infos)
        self.current_ids[start_index:self.storage_filled_to_index[-1]] = insert_idx
        self.doc_infos[insert_idx] = (self.token_base_number,start_index,self.storage_filled_to_index[-1])
        return insert_idx

    def lookup(self, idx:str):
        '''
        ids: need to be int64
        '''
        doc_inf = self.doc_infos[idx]
        return self.storage[doc_inf[0]][doc_inf[1]:doc_inf[2]]

    def save(self):
        '''
        save the data to disk
        '''
        saveCompressed(os.path.join(self.run_folder,self.name+"_mappings.npz"),
                       doc_infos=self.doc_infos,id_mapping=self.id_mapping,
                       storage_filled_to_index=self.storage_filled_to_index)

    def get_for_indexing(self):
        '''
        get all vectors for indexing
        '''
        storage_view = []
        mapping_view = []
        for i,s in enumerate(self.storage):
            storage_view.append(s[:self.storage_filled_to_index[i]])
            mapping_view.append(self.id_mapping[i][:self.storage_filled_to_index[i]])
        return mapping_view,storage_view

    def load(self, folder:str):
        '''
        load a saved vector-storage
        '''
        dfs = numpy.load(os.path.join(folder,self.name+"_mappings.npz"),allow_pickle=True)
        self.doc_infos=dfs.get("doc_infos")[()]
        self.id_mapping=dfs.get("id_mapping")[()]
        self.storage_filled_to_index=dfs.get("storage_filled_to_index")[()]

        self.storage = []
        for f in range(0,len(glob.glob(os.path.join(folder,self.name+"_storage_*")))):
            self.storage.append(numpy.memmap(os.path.join(folder,self.name+"_storage_"+str(f)+".npy"), dtype=numpy.dtype(self.token_dtype),
                                        mode="r", shape=(self.token_base_size,self.vector_dimensions)))

    def print_statistics(self):

        total_vectors = sum(self.storage_filled_to_index)

        print(self.name+" vector storage statistics:")
        print("\tTotal vectors:",total_vectors)
        print("\tDimensions:",self.vector_dimensions)
        bytes_per_dim = -1
        if self.token_dtype == "float16": bytes_per_dim = 2
        elif self.token_dtype == "float32": bytes_per_dim = 4
        elif self.token_dtype == "int32": bytes_per_dim = 4
        elif self.token_dtype == "uint32": bytes_per_dim = 4
        elif self.token_dtype == "int64": bytes_per_dim = 8
        print("\tTotal storage:",total_vectors*self.vector_dimensions*bytes_per_dim/1024/1024,"MB")
        print("\tTotal documents:",len(self.doc_infos))
        print("\tAvg. vectors / document:",total_vectors/len(self.doc_infos))
        print("\tAvg. bytes per document:",total_vectors*self.vector_dimensions*bytes_per_dim/len(self.doc_infos)," bytes")
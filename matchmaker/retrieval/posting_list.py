from typing import Any, Dict, Iterator, List
import numpy

class PostingList():
    '''
    Dynamic memory mapped + self growing posting list 
    '''

    def __init__(self, dtype, dimensions):
        super(PostingList, self).__init__()
        self.vector_dimensions = dimensions
        self.token_dtype = dtype

        self.data_values = numpy.zeros((1,self.vector_dimensions),dtype=self.token_dtype)
        self.data_ids = numpy.zeros((1,),dtype='int32')
        self.capacity = 1
        self.size = 0


    def add(self, vector:numpy.ndarray, id:int):
        '''
        Add either a vector to the storage
        '''
        #
        # 2 options: 1) first add, 2) current store is full
        #
        if self.size == self.capacity:
            self.capacity *= 2
            
            newdata = numpy.zeros((self.capacity,self.vector_dimensions),dtype=self.token_dtype)
            newdata[:self.size] = self.data_values
            self.data_values = newdata

            newdata = numpy.zeros((self.capacity,),dtype='int32')
            newdata[:self.size] = self.data_ids
            self.data_ids = newdata

        self.data_values[self.size] = vector
        self.data_ids[self.size] = id
        self.size += 1
        

    def get(self):
        '''
        view of stored data and ids 
        '''
        return (self.data_values[:self.size], self.data_ids[:self.size])

    def get_bytes(self):

        bytes_per_dim = -1
        if self.token_dtype == "float16": bytes_per_dim = 2
        elif self.token_dtype == "float32": bytes_per_dim = 4
        elif self.token_dtype == "int32": bytes_per_dim = 4
        elif self.token_dtype == "uint32": bytes_per_dim = 4
        elif self.token_dtype == "int64": bytes_per_dim = 8

        return self.size*self.vector_dimensions*bytes_per_dim + self.size*4

from typing import Any, Dict, List, Optional, Union

import copy
import random
import numpy
import torch
from allennlp.data.fields import TransformerTextField
from allennlp.data.tokenizers import Token, Tokenizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from transformers import (AutoTokenizer, BartTokenizer,
                          PreTrainedTokenizerFast, T5Tokenizer)
import nltk
nltk.download("punkt", quiet=True)

import hashlib

class CustomTransformerTextField(TransformerTextField):

    def __init__(self,input_ids: Union[torch.Tensor, List[int]],
                      attention_mask: Optional[Union[torch.Tensor, List[int]]] = None,
                      unique_input_ids: Optional[Union[torch.Tensor, List[int]]] = None,
                      input_ids_to_words_map=None,
                      unique_words=None,
                      padding_token_id: int = 0):

        super().__init__(input_ids=input_ids, attention_mask=attention_mask, padding_token_id=padding_token_id)
        if unique_input_ids is not None:
            self.unique_input_ids = unique_input_ids
            if len(self.__slots__)==6:
                self.__slots__.append("unique_input_ids")
        if input_ids_to_words_map is not None:
            self.input_ids_to_words_map = input_ids_to_words_map
            if len(self.__slots__)==7:
                self.__slots__.append("input_ids_to_words_map")

        if unique_words is not None:
            self.unique_words = unique_words
            if len(self.__slots__)==8:
                self.__slots__.append("unique_words")
        


class FastTransformerTokenizer():
    """
    basic wrapper for an HuggingFace AutoTokenizer
    """

    def __init__(self, model,add_unique_ids=False,uniqueness_type="lower",create_global_id=False):

        if "t5" in model:
            self._tokenizer = T5Tokenizer.from_pretrained(model)
            # when generating, we will use the logits of right-most token to predict the next token
            # so the padding should be on the left
            self._tokenizer.padding_side = "left"
            self._tokenizer.pad_token = self._tokenizer.eos_token # to avoid an error
        elif "bart" in model:
            self._tokenizer = BartTokenizer.from_pretrained(model)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model)

        self.add_unique_ids = add_unique_ids
        if self.add_unique_ids:
            self.pre_tokenzier = BertPreTokenizer()

            from nltk.stem.porter import PorterStemmer
            self.stemmer = PorterStemmer()
            
            self.uniqueness_type = uniqueness_type # or "stemmed"
            self.create_global_id = create_global_id

            self.stem_cache = {}

    def tokenize(self, sentence: str, sentence2: str = None, max_length: int = 512, padding=False, random_spans=False):
        if sentence2 != None:
            seq_tokenized = self._tokenizer(sentence, sentence2,
                                            max_length=max_length,
                                            truncation=True,
                                            return_tensors="pt",
                                            return_attention_mask=True,
                                            padding="max_length" if padding else False)

        else:
            if random_spans:
                sentences = nltk.sent_tokenize(sentence)
                sentence_ids = list(range(len(sentences)))
                random.shuffle(sentence_ids)
                sent_length = 0
                sentence = ''
                for id in sentence_ids:
                    sent = sentences[id]
                    if len(sent.split(' ')) + sent_length < 512:
                        sentence = sentence + sent
                        sent_length = len(sent.split(' '))

                seq_tokenized = self._tokenizer(sentence,
                                                max_length=max_length,
                                                truncation=True,
                                                return_tensors="pt",
                                                return_attention_mask=True,
                                                padding="max_length" if padding else False)
            else:
                seq_tokenized = self._tokenizer(sentence,
                                                max_length=max_length,
                                                truncation=True,
                                                return_tensors="pt",
                                                return_attention_mask=True,
                                                padding="max_length" if padding else False)

            #
            # only used for ColBERTer model
            #
            if self.add_unique_ids:

                seq_tokenized.data["unique_input_ids"] = torch.unique(seq_tokenized.data["input_ids"])
                
                # these are the wordpiece-subwords
                tf_offsets = seq_tokenized.encodings[0].offsets

                # these are the whole-word offsets (subwords are not split yet), but it uses the exact same splitting mechanism
                whole_word_offsets = self.pre_tokenzier.pre_tokenize_str(sentence)

                # create unique_token_dict
                whole_word_unique = {}
                for i,(tok,offsets) in enumerate(whole_word_offsets):
                    if self.uniqueness_type == "stemmed":
                        lower_tok = tok.lower()
                        if lower_tok not in self.stem_cache:
                            tok_transformed = self.stemmer.stem(lower_tok)
                            self.stem_cache[lower_tok] = tok_transformed
                        else:
                            tok_transformed = self.stem_cache[lower_tok]
                    else:
                        tok_transformed = tok.lower()

                    whole_word_offsets[i] = (tok_transformed,offsets)
                    
                    if tok_transformed not in whole_word_unique:
                        if self.create_global_id:
                            hashed = int.from_bytes(hashlib.sha256(tok_transformed.encode('utf-8')).digest()[:4], 'little', signed=False) # 32-bit int
                            # 0 is a reserved id for padding, don't think this will happen often though
                            if hashed == 0:
                                hashed = 1
                                
                            if hashed < 0 or hashed > 4294967295:
                            #if hashed < -2147483648 or hashed > 2147483647:
                                print("Warning: hash value is too large, will be truncated to 32-bit int")
                            whole_word_unique[tok_transformed] = hashed
                        else:
                            whole_word_unique[tok_transformed] = len(whole_word_unique) + 1

                # map tf_offsets to whole_word_unique
                tf_input_ids_to_whole_word_unique_map = torch.zeros_like(seq_tokenized.data["input_ids"])
                for i,tf_offset in enumerate(tf_offsets[1:-1]): # ignore special tokens
                    for whole_word_token,whole_word_offset in whole_word_offsets:
                        if tf_offset[0] >= whole_word_offset[0] and tf_offset[1] <= whole_word_offset[1]:
                            tf_input_ids_to_whole_word_unique_map[0][i+1] = whole_word_unique[whole_word_token]
                            break
                
                # if the tokenizer cuts off the sequence, we might have some tokens that are in the pre-tokenizer, but not mapped
                # because they only appear in the end and where cut -> in this case we just remove them also from the unique list
                # as the main tokenizer is the main anchor point
                skipped_whole_word =[]
                for tok,i in whole_word_unique.items():
                    if i not in tf_input_ids_to_whole_word_unique_map[0]:
                        skipped_whole_word.append(tok)
                for tok in skipped_whole_word:
                    del whole_word_unique[tok]

                #
                # this is just sanity checking to make sure that the mapping is correct
                #
                #if (tf_input_ids_to_whole_word_unique_map[0][1:-1] == 0).any():
                #    missing_ids = seq_tokenized.data["input_ids"][0][1:-1][tf_input_ids_to_whole_word_unique_map[0][1:-1] == 0]
                #    missing_toks = self._tokenizer.convert_ids_to_tokens(missing_ids)
                #    if not (len(set(missing_toks)) <= 2 and ((set(missing_toks) == set(["[PAD]", "[SEP]"])) or missing_toks[0] == "[PAD]")):
                #        print("WARNING: some tokens were not found in the whole_word dictionary",missing_toks,"in sentence:", sentence, "with offset:", whole_word_offsets,"unique_words", whole_word_unique)

                seq_tokenized.data["input_ids_to_words_map"] = tf_input_ids_to_whole_word_unique_map
                seq_tokenized.data["unique_words"] = torch.from_numpy(numpy.array(list(whole_word_unique.values()),dtype=numpy.int64)).unsqueeze(0)

        for _, d in seq_tokenized.data.items():
            d.squeeze_(0)
        return seq_tokenized.data

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/corpus.py
"""

from collections import namedtuple
import os
import torch
from copy import deepcopy

from tqdm import tqdm
from source.inputters.field import tokenize
from source.inputters.field import TextField
from source.inputters.field import NumberField
from source.inputters.dataset import Dataset

class Corpus(object):
    """
    Corpus
    """

    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 vocab_size=None,
                 tokenizer=None):
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        self.min_freq = min_freq
        self.vocab_size = vocab_size

        prepared_data_file = data_prefix + "_" + \
            str(vocab_size) + ".data.pt"
        prepared_vocab_file = data_prefix + "_" + \
            str(vocab_size) + ".vocab.pt"

        self.prepared_data_file = os.path.join(data_dir, prepared_data_file)
        self.prepared_vocab_file = os.path.join(data_dir, prepared_vocab_file)
        self.fields = {}
        self.filter_pred = None
        self.sort_fn = None
        self.data = None
        self.tokenizer = tokenizer
        self.train_sampler = None

    def load(self):
        """
        load
        """
        if self.tokenizer is not None:
            if not os.path.exists(self.prepared_data_file):
                self.build()
            self.load_data(self.prepared_data_file)
        else:
            if not (os.path.exists(self.prepared_data_file) and
                    os.path.exists(self.prepared_vocab_file)):
                self.build()
            self.load_vocab(self.prepared_vocab_file)
            self.load_data(self.prepared_data_file)

            self.padding_idx = self.TGT.stoi[self.TGT.pad_token]

    def reload(self, data_type='test'):
        """
        reload
        """
        data_file = os.path.join(
            self.data_dir, self.data_prefix + "." + data_type)
        data_raw = self.read_data(data_file, data_type="test")
        data_examples = self.build_examples(data_raw)
        self.data[data_type] = Dataset(data_examples)

        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_data(self, prepared_data_file=None):
        """
        load_data
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        data = torch.load(prepared_data_file)
        self.data = {"train": Dataset(data['train']),
                     "valid": Dataset(data["valid"]),
                     "test": Dataset(data["test"])}
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_vocab(self, prepared_vocab_file):
        """
        load_vocab
        """
        prepared_vocab_file = prepared_vocab_file or self.prepared_vocab_file
        print("Loading prepared vocab from {} ...".format(prepared_vocab_file))
        vocab_dict = torch.load(prepared_vocab_file)

        for name, vocab in vocab_dict.items():
            if name in self.fields:
                self.fields[name].load_vocab(vocab)
        print("Vocabulary size of fields:",
              " ".join("{}-{}".format(name.upper(), field.vocab_size)
                       for name, field in self.fields.items()
                       if isinstance(field, TextField)))

    def read_data(self, data_file, data_type=None):
        """
        Returns
        -------
        data: ``List[Dict]``
        """
        raise NotImplementedError

    def build_vocab(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        field_data_dict = {}
        for name in ['src', 'tgt', 'cue']:
            field = self.fields.get(name)
            if isinstance(field, TextField):
                xs = [x[name] for x in data]
                if field not in field_data_dict:
                    field_data_dict[field] = xs
                else:
                    field_data_dict[field] += xs

        vocab_dict = {}
        for name, field in self.fields.items():
            if field in field_data_dict:
                print("Building vocabulary of field {} ...".format(name.upper()))
                if field.vocab_size == 0:
                    field.build_vocab(field_data_dict[field],
                                      min_freq=self.min_freq,
                                      max_size=self.vocab_size)
                vocab_dict[name] = field.dump_vocab()
        return vocab_dict

    def build_examples(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        examples = []
        for raw_data in tqdm(data):
            example = {}
            for name, strings in raw_data.items():
                example[name] = self.fields[name].numericalize(strings)
            examples.append(example)
        if self.sort_fn is not None:
            print("Sorting examples ...")
            examples = self.sort_fn(examples)
        return examples

    def build(self):
        """
        build
        """
        print("Start to build corpus!")
        train_file = os.path.join(self.data_dir, "train.txt")
        valid_file = os.path.join(self.data_dir, "valid.txt")
        test_file = os.path.join(self.data_dir, "test.txt")

        print("Reading data ...")
        train_raw = self.read_data(train_file, data_type="train")
        valid_raw = self.read_data(valid_file, data_type="valid")
        test_raw = self.read_data(test_file, data_type="test")

        if self.tokenizer is None:
            vocab = self.build_vocab(train_raw)
            print("Saving prepared vocab ...")
            torch.save(vocab, self.prepared_vocab_file)
            print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))

        print("Building TRAIN examples ...")
        train_data = self.build_examples(train_raw)
        print("Building VALID examples ...")
        valid_data = self.build_examples(valid_raw)
        print("Building TEST examples ...")
        test_data = self.build_examples(test_raw)

        data = {"train": train_data,
                "valid": valid_data,
                "test": test_data}

        print("Saving prepared data ...")
        torch.save(data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))

    def create_dataloader(self, batch_size, data_type="train",
                       shuffle=False, device=None, parallel=False):
        """
        create_dataloader
        """
        try:
            data = self.data[data_type]
            if data_type == 'train' and parallel:
                data_sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=shuffle)
                data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=data_sampler, collate_fn=Dataset.collate_fn(device))
                self.train_sampler = data_sampler
            else:
                data_loader = data.create_dataloader(batch_size, shuffle, device)
            return data_loader
        except KeyError:
            raise KeyError("Unsported data type: {}!".format(data_type))

    def transform(self, data_file, batch_size,
                  data_type="test", shuffle=False, device=None):
        """
        Transform raw text from data_file to Dataset and create data loader.
        """
        raw_data = self.read_data(data_file, data_type=data_type)
        examples = self.build_examples(raw_data)
        data = Dataset(examples)
        data_loader = data.create_dataloader(batch_size, shuffle, device)
        return data_loader


class KnowledgeCorpus(Corpus):
    """
    KnowledgeCorpus
    """

    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 vocab_size=None,
                 min_len=0,
                 max_len=100,
                 embed_file=None,
                 share_vocab=False,
                 with_label=False,
                 tokenizer=None):
        super(KnowledgeCorpus, self).__init__(data_dir=data_dir,
                                                     data_prefix=data_prefix,
                                                     min_freq=min_freq,
                                                     vocab_size=vocab_size,
                                                     tokenizer=tokenizer)
        self.min_len = min_len
        self.max_len = max_len
        self.share_vocab = share_vocab
        self.with_label = with_label
        self.num_samples = {'train':0, 'valid':0, 'test':0}

        self.SRC = TextField(tokenize_fn=tokenize,
                             embed_file=embed_file,
                             tokenizer=self.tokenizer)
        if self.share_vocab:
            self.TGT = self.SRC
            self.CUE = self.SRC
        else:
            self.TGT = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file,
                                 tokenizer=self.tokenizer)
            self.CUE = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file,
                                 tokenizer=self.tokenizer)

        if self.with_label:
            self.INDEX = NumberField()
            self.fields = {'src': self.SRC, 'tgt': self.TGT,
                           'cue': self.CUE, 'index': self.INDEX}
        else:
            self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE}

        def src_filter_pred(src):
            """
            src_filter_pred
            """
            return min_len <= len(self.SRC.tokenize_fn(src)) <= max_len

        def tgt_filter_pred(tgt):
            """
            tgt_filter_pred
            """
            return min_len <= len(self.TGT.tokenize_fn(tgt)) <= max_len

        self.filter_pred = lambda ex: src_filter_pred(
            ex['src']) and tgt_filter_pred(ex['tgt'])

    def read_data(self, data_file, data_type="train"):
        """
        read_data
        """
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                if self.with_label:
                    src, tgt, knowledge, label = line.strip().split('\t')[:4]
                else:
                    src, tgt, knowledge = line.strip().split('\t')[:3]
                filter_src = []
                filter_tgt = []
                filter_knowledge = []
                filter_label = []
                for src_sent, tgt_sent in zip(src.split('\x01'), tgt.split('\x01')):
                    src_tokens = tokenize(src_sent)
                    tgt_tokens = tokenize(tgt_sent)
                    if len(src_tokens) <= self.max_len and len(tgt_tokens) <= self.max_len:
                        filter_src.append(' '.join(src_tokens))
                        filter_tgt.append(' '.join(tgt_tokens))
                for sent in knowledge.split('\x01'):
                    filter_knowledge.append(
                        ' '.join(tokenize(sent)[:self.max_len]))
                if len(filter_knowledge) == 0:
                    continue
                if self.with_label:
                    for sent in label.split('\x01'):
                        filter_label.append(sent)
                data.append({'src': filter_src, 'tgt': filter_tgt, 'cue': filter_knowledge})
                self.num_samples[data_type] += len(filter_src)


        filtered_num = len(data)
        filtered_num -= len(data)
        print(
            "Read {} {} examples ({} filtered)".format(len(data), data_type.upper(), filtered_num))
        return data

    def build_examples(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        examples = []
        for raw_data in tqdm(data):
            example = {}
            for name, strings in raw_data.items():
                example[name] = [self.fields[name].numericalize(x) for x in strings]
            examples.append(example)
        if self.sort_fn is not None:
            print("Sorting examples ...")
            examples = self.sort_fn(examples)
        return examples

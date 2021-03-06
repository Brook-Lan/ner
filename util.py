#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2019-02-21

@author:Brook
"""
import os
import re
import json
from collections import namedtuple

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from corpus import corpus as _corpus


BASE_DIR = os.path.dirname(__file__)
VOCAB_PATH = os.path.join(BASE_DIR, "data/vocab.json")

TAGS = ("O", 
        "B-Level", "I-Level",
        "B-Reason", "I-Reason", 
        "B-SideEff", "I-SideEff", 
        "B-Symptom", "I-Symptom", 
        "B-Disease", "I-Disease", 
        "B-Operation", "I-Operation", 
        "B-Drug", "I-Drug",
        "B-Test_Value", "I-Test_Value",
        "B-Duration", "I-Duration", 
        "B-Amount", "I-Amount", 
        "B-Treatment", "I-Treatment", 
        "B-Frequency", "I-Frequency", 
        "B-Anatomy",  "I-Anatomy", 
        "B-Test", "I-Test",
        "B-Method", "I-Method", 
        )


def build_vocab():
    """构造"词汇 -> id"映射
    """
    word_count = {}
    for chars, _ in _corpus:
        for char in chars:
            if re.search("\d", char):
                char = "<NUM>"
            if char in word_count:
                word_count[char] += 1
            else:
                word_count[char] = 1
    words = [w for w,f in word_count.items() if f > 2]
    # 每个词映射唯一id
    word2id = {w:i for i, w in enumerate(words,3)}
    word2id['<PAD>'] = 0   # padding
    word2id['<SOS>'] = 1   # start of sequence
    word2id['<UNK>'] = 2   # unknown
    # 序列化为JSON
    with open(VOCAB_PATH, 'w') as f:
        json.dump(word2id, f)


def load_vocab():
    with open(VOCAB_PATH) as f:
        word2id = json.load(f)
        return word2id


class IdMapper:
    """id映射的类方法
    - 将词的列拜映射为对应的id的列表
    - 将标注映射为对应的标注id
    """
    WORD2ID = load_vocab()
    TAG2ID = {tag:i for i, tag in enumerate(TAGS)}

    @classmethod
    def encode_sequences(cls, sequences):
        """词序列转换到id序列
        """
        word2id = cls.WORD2ID
        new_sequences = []
        for seq in sequences:
            ids = []
            for w in seq:
                if re.search("\d", w):
                    w = '<NUM>'
                if w not in word2id:
                    w = '<UNK>'
                ids.append(word2id[w])
            new_sequences.append(ids)
        return new_sequences

    @classmethod
    def encode_tags(cls, tags_list):
        """编码标注序列为id
        """
        tag2id = cls.TAG2ID
        new_tags = []
        for tags in tags_list:
            ids = []
            for tag in tags:
                ids.append(tag2id[tag])
            new_tags.append(ids)
        return new_tags

    @classmethod
    def decode_tags(cls, tags_list):
        """将标注id还原为标注
        """
        id2tag = {v:k for k,v in cls.tag2id.items()}
        new_tags = []
        for tags in tags_list:
            ids = []
            for tag in tags:
                ids.append(id2tag[tag])
            new_tags.append(ids)
        return new_tags


def encode_docs(docs, max_len=500):
    x_train = IdMapper.encode_sequences(docs)
    x_train = pad_sequences(x_train, max_len)
    return x_train


def encode_tags(tags, max_len=500):
    y_train = IdMapper.encode_tags(tags)
    y_train = pad_sequences(y_train, max_len)
    y_train = np.expand_dims(y_train, axis=-1)
    return y_train


EntityIndex = namedtuple("Entity_Index",["entity_type", "begin", "end"])
Entity = namedtuple("Entity", ["entity_type", "entity"])

class Result:
    """模型输出结果的封装
    """
    _TAGS = TAGS

    def __init__(self, raw_output, input_data, padding_method=None):
        self.raw_output = raw_output
        self.input_data = input_data
        self.padding_method = padding_method

    @classmethod
    def recover_tags(cls, output):
        """还原成tag序列
        """
        indexes_list = [np.argmax(a_output, axis=-1) for a_output in output]
        tags_list = [[cls._TAGS[ind] for ind in indexes] for indexes in indexes_list]
        return tags_list

    @classmethod
    def remove_padding(cls, raw_output, input_data):
        """去除padding
        """
        output = []
        for a_output, a_input in zip(raw_output, input_data):
            output.append(a_output[-len(a_input):])
        return output
  
    @classmethod
    def cal_entity_index(cls, tags_list):
        """计算实体位置
        """
        def _cal_entity_index(tags):
            tag_ind = []
            cur_tag = "o"
            type_ = "non"
            start = 0
            for i, tag in enumerate(tags):
                if tag == "O":
                    if cur_tag == "i":
                        tag_ind.append(EntityIndex(type_, start, i))
                    start = i
                    cur_tag = 'o'
                    
                elif tag.startswith("B-"):
                    if cur_tag == "i":
                        tag_ind.append(EntityIndex(type_, start, i))
                    start = i
                    cur_tag = "b"
                    type_ = tag.replace("B-", "")
                    
                elif tag.startswith("I-"):
                    cur_tag = 'i'
                else:
                    raise ValueError("unknown tag: %s" % tag)
            return tag_ind

        indexes = []
        for tags in tags_list:
            index = _cal_entity_index(tags)
            indexes.append(index)
        return indexes

    @property
    def entities(self):
        raw_output, input_data = self.raw_output, self.input_data
        output = self.remove_padding(raw_output, input_data)
        tags_list = self.recover_tags(output)
        entity_indexes = self.cal_entity_index(tags_list)
        results = []
        for doc, indexes in zip(input_data, entity_indexes):
            entities = []
            for index in indexes:
                ent = doc[index.begin:index.end]
                entity = Entity(index.entity_type, ent)
                entities.append(entity)
            results.append(entities)
        return results


if __name__ == "__main__":
    build_vocab()



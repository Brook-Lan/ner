#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2019-02-21

@author:Brook
"""
import os
import re
import json

import numpy as np

from corpus import corpus as _corpus


BASE_DIR = os.path.dirname(__file__)
VOCAB_PATH = os.path.join(BASE_DIR, "data/vocab.json")
CONST_PATH = os.path.join(BASE_DIR, "data/const.json")

TAGS = ("I-Test", "I-Method", "B-Level", "I-Drug", "B-Reason", "I-Operation", "I-SideEff", "B-SideEff", "I-Reason", "B-Symptom", "I-Test_Value", "I-Anatomy", "B-Disease", "B-Operation", "I-Duration", "I-Disease", "B-Drug", "B-Test_Value", "I-Amount", "I-Level", "B-Duration", "B-Amount", "I-Treatment", "I-Symptom", "B-Treatment", "I-Frequency", "B-Frequency", "B-Anatomy", "B-Test", "B-Method", "O")


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


class IdMapper:
    def __init__(self):
        with open(VOCAH_PATH) as f:
            word2id = json.load(f)
        self.word2id = word2id
        self.tag2id = {tag:i for i, tag in enumerate(TAGS)}

    def encode_sequences(self, sequences):
        """词序列转换到id序列
        """
        word2id = self.word2id
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
        return np.array(new_sequences)

    def encode_tags(self, tags_list):
        """编码标注序列为id
        """
        tag2id = self.tag2id
        new_tags = []
        for tags in tags_list:
            ids = []
            for tag in tags:
                ids.append(tag2id[tag])
            new_tags.append(ids)
        return np.array(new_tags)

    def decode_tags(self, tags_list):
        """将标注id还原为标注
        """
        id2tag = {v:k for k,v in self.tag2id.items()}
        new_tags = []
        for tags in tags_list:
            ids = []
            for tag in tags:
                ids.append(id2tag[tag])
            new_tags.append(ids)
        return new_tags



if __name__ == "__main__":
    build_vocab()

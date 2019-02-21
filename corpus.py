#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2019-02-21

@author:Brook
"""
import os
import re
import csv

BASE_DIR = os.path.dirname(__file__)
CORPUS_PATH = os.path.join(BASE_DIR, "data/train")


class Corpus:
    def __init__(self, corpus_path=None):
        if corpus_path is None:
            corpus_path = CORPUS_PATH
        self.corpus_path = corpus_path

    def _load_corpus(self, path):
        with open(path) as f:
            reader = csv.reader(f,delimiter="\t")
            for char, tag in reader:
                # 去掉空白
                if char.strip() == "":
                    continue
                yield char, tag

    def load_corpus(self, path):
        if os.path.isfile(path):
            chars, tags = zip(*self._load_corpus(path))
            #yield chars, tags
            ## 分割文章
            sent = "".join(chars)
            previous_end = 0
            for m in re.finditer("。", sent):
                begin, end = m.span()
                s = chars[previous_end:begin+1]
                t = tags[previous_end:begin+1]
                previous_end = end
                yield s, t
            # 如果全文没有分隔符，返回原数据
            if previous_end == 0:
                yield chars, tags


        elif os.path.isdir(path):
            for fname in os.listdir(path):
                fpath = os.path.join(path, fname)
                for chars, tags in self.load_corpus(fpath):
                    yield chars, tags

    def __iter__(self):
        return self.load_corpus(self.corpus_path)

 
corpus = Corpus()


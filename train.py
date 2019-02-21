#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2019-02-21

@author:Brook
"""
from keras

from corpus import corpus


if __name__ == "__main__":
    cs,ts = next(corpus())
    word_counts = Counter(cs)
    vocab = [w for w,f in word_counts.items() if f >=2]

    print(word_counts.most_common())


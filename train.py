#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2019-02-21

@author:Brook
"""
import numpy as np

from corpus import corpus
from util import encode_docs, encode_tags
from model import build_bilstm_crf_model, save_model


if __name__ == "__main__":
    MAX_LEN = 500

    x_train, y_train = zip(*corpus)
    x_train = encode_docs(x_train, MAX_LEN)
    y_train = encode_tags(y_train, MAX_LEN)

    model = build_bilstm_crf_model()

    history = model.fit(x_train, 
                        y_train,
                        batch_size=30,
                        epochs=5, 
                        validation_split=0.8)

    save_model(model, "model2.h5")


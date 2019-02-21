#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2019-02-21

@author:Brook
"""
import json

from keras.models import Sequential
from keras.layers import Embedding, LSTM,Bidirectional, Dense, TimeDistributed, Dropout
from keras_contrib.layers.crf import CRF
from keras_contrib.utils import save_load_utils


EMBEDDING_OUT_DIM = 128
TIME_STAMPS = 100
HIDDEN_UNITS = 200
DROPOUT_RATE = 0.3

with open("data/const.json") as f:
    const = json.load(f)

VOCAB_SIZE =  const['VOCAB_SIZE']
NUM_CLASS =  len(const['TAGS'])


def build_bilstm_crf_model():
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, output_dim=EMBEDDING_OUT_DIM, input_length=TIME_STAMPS))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(TimeDistributed(Dense(NUM_CLASS)))
    
    crf_layer = CRF(NUM_CLASS)
    model.add(crf_layer)
    model.compile("rmsprop", loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
    return model


def save_bilstm_crf_model(mode, filename):
    save_load_utils.save_all_weights(model, filename)


def load_bilstm_crf_model(filename):
    model = build_bilstm_crf_model()
    save_load_utils.load_all_weights(mode, filename)
    return model


if __name__ == "__main__":
    model = build_bilstm_crf_model()
    model.summary()

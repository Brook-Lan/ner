#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2019-02-21

@author:Brook
"""
from keras.models import Sequential
from keras.layers import Embedding, LSTM,Bidirectional, Dense, TimeDistributed, Dropout
from keras_contrib.layers.crf import CRF
from keras_contrib.utils import save_load_utils

from util import load_vocab, TAGS
word2id = load_vocab()


VOCAB_SIZE = len(word2id) 
NUM_CLASS =  len(TAGS)

EMBEDDING_OUT_DIM = 128
TIME_STAMPS = 100
HIDDEN_UNITS = 200
DROPOUT_RATE = 0.3

EMBEDDING_INPUT_LEN = 500


def build_bilstm_crf_model():
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, output_dim=EMBEDDING_OUT_DIM, input_length=EMBEDDING_INPUT_LEN))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    #model.add(Dropout(DROPOUT_RATE))
    #model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    #model.add(Dropout(DROPOUT_RATE))
    model.add(TimeDistributed(Dense(NUM_CLASS)))
    crf = CRF(NUM_CLASS, sparse_target=True)
    model.add(crf)
    model.compile("adam", loss=crf.loss_function, metrics=[crf.accuracy])
    return model


def save_bilstm_crf_model(model, filename):
    save_load_utils.save_all_weights(model, filename)


def load_bilstm_crf_model(filename):
    model = build_bilstm_crf_model()
    save_load_utils.load_all_weights(model, filename)
    return model


if __name__ == "__main__":
    model = build_bilstm_crf_model()
    model.summary()

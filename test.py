#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@create Time:2019-02-22

@author:Brook
"""
import re

from model import load_model
from util import encode_docs, Result


if __name__ == "__main__":
    model = load_model(None, False)

    path = "data/test/7.txt"
    with open(path) as f:
        text = f.read()
    text = re.sub("\s", "", text)
    docs = text.split("。")
    x_test = encode_docs(docs)

    y_pred = model.predict(x_test)
    result = Result(y_pred, docs)
    print(result.entities) 



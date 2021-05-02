# -*- coding:utf-8 -*-

import tensorflow as tf


def preprocessing_fn(inputs):
    outputs = {}

    outputs["image_tf"] = inputs["image"]
    outputs["label_tf"] = inputs["label"]
    return outputs

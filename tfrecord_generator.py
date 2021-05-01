# -*- coding:utf-8 -*-

import os
import glob

import numpy as np
import tensorflow as tf
from PIL import Image

DATASET_DIR_PATH = "./data"


def create_tfrecord(img_files, out_file):
    # create output directory
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # create tfrecord
    with tf.io.TFRecordWriter(out_file) as writer:
        for f in img_files:
            with Image.open(f) as img_obj:
                image = np.array(img_obj)
                width = image.shape[0]
                height = image.shape[1]
                image_raw = image.tostring()
                label = 0 if "cat" in f else 1

                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_obj.tobytes()]))
                    })
                )
            writer.write(example.SerializeToString())


train_files = glob.glob("original_data/training_set/training_set/*/*.jpg")
test_files = glob.glob("original_data/test_set/test_set/*/*.jpg")

create_tfrecord(train_files, os.path.join(
    DATASET_DIR_PATH, "train/dog_cat_train.tfrecord"))
create_tfrecord(test_files, os.path.join(
    DATASET_DIR_PATH, "test/dog_cat_test.tfrecord"))

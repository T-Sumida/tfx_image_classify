# -*- coding:utf-8 -*-

import os
import io
import base64
import glob

import numpy as np
import tensorflow as tf
from PIL import Image

DATASET_DIR_PATH = "./data"


def create_tfrecord(img_files, out_file, file_limit=100):
    # create output directory
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    dog_counter = 0
    cat_counter = 0

    # create tfrecord
    with tf.io.TFRecordWriter(out_file) as writer:
        for f in img_files:
            with Image.open(f) as img_obj:
                img_obj = img_obj.convert('RGB').resize((500, 500))
                image = np.array(img_obj)
                width = image.shape[0]
                height = image.shape[1]

                img_bytes = io.BytesIO()
                img_obj.save(img_bytes, format='JPEG')
                img_bytes = base64.urlsafe_b64encode(img_bytes.getvalue())

                label = 0 if "dogs" in f else 1

                if dog_counter == file_limit and label == 0:
                    continue
                if cat_counter == file_limit and label == 1:
                    continue
                if dog_counter == file_limit and cat_counter == file_limit:
                    break

                if label == 0:
                    dog_counter += 1
                else:
                    cat_counter += 1

                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))
                    })
                )
            writer.write(example.SerializeToString())


train_files = glob.glob("original_data/training_set/training_set/*/*.jpg")
test_files = glob.glob("original_data/test_set/test_set/*/*.jpg")

create_tfrecord(train_files, os.path.join(
    DATASET_DIR_PATH, "train/dog_cat_train.tfrecord"), file_limit=500)
create_tfrecord(test_files, os.path.join(
    DATASET_DIR_PATH, "test/dog_cat_test.tfrecord"))


cnt = len(list(tf.compat.v1.io.tf_record_iterator(os.path.join(
    DATASET_DIR_PATH, "train/dog_cat_train.tfrecord"))))
print("Train データ件数：{}".format(cnt))

cnt = len(list(tf.compat.v1.io.tf_record_iterator(os.path.join(
    DATASET_DIR_PATH, "test/dog_cat_test.tfrecord"))))
print("Test  データ件数：{}".format(cnt))

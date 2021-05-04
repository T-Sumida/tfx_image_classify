# -*- coding:utf-8 -*-
import absl
from typing import List, Text

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options

absl.logging._warn_preinit_stderr = False

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 1e-3
IMG_SIZE = 224
IMG_KEY = "image"
LABEL_KEY = 'label'


def _transform_key_name(key):
    return key + '_xf'


def _make_serving_signatures(model: tf.keras.Model, tf_transform_features: tft.TFTransformOutput):
    model.tft_layer = tf_transform_features.transform_features_layer()

    @tf.function
    def serve_image_fn(image_byte):
        feature_spec = tf_transform_features.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(image_byte, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return {
        'serving_default':
        serve_image_fn.get_concrete_function(
            tf.TensorSpec(
                shape=[None], dtype=tf.string, name='examples'))
    }


def _create_dataset(
    file_pattern: List[Text],
    data_accessor: DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    is_train: bool = False,
    batch_size: int = 200
) -> tf.data.Dataset:

    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_transform_key_name(LABEL_KEY)
        ),
        tf_transform_output.transformed_metadata.schema
    )
    return dataset


def _build_model() -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        include_top=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'
    )
    feature = base_model.output
    x = tf.keras.layers.Dropout(0.1)(feature)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
        metrics=['sparse_categorical_accuracy']
    )
    model.summary(print_fn=absl.logging.info)
    return model


def preprocessing_fn(inputs):
    outputs = {}

    decoded = tf.io.decode_base64(inputs['image'])
    images = tf.map_fn(
        lambda x: tf.io.decode_jpeg(x[0], channels=3),
        decoded, dtype=tf.uint8
    )

    images = tf.cast(images, tf.float32)
    images = tf.image.resize(images, [IMG_SIZE, IMG_SIZE])

    outputs[_transform_key_name(IMG_KEY)] = images
    outputs[_transform_key_name(LABEL_KEY)] = inputs[LABEL_KEY]
    return outputs


def run_fn(fn_args: TrainerFnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _create_dataset(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=True,
        batch_size=TRAIN_BATCH_SIZE
    )
    eval_dataset = _create_dataset(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=False,
        batch_size=TRAIN_BATCH_SIZE
    )

    model = _build_model()

    absl.logging.info(
        'Tensorboard logging to {}'.format(fn_args.model_run_dir))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')

    steps_per_epoch = int(100 / TRAIN_BATCH_SIZE)
    absl.logging.info('Start training the top classifier')
    model.fit(
        train_dataset,
        epochs=2,
        steps_per_epoch=steps_per_epoch,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])

    signatures = _make_serving_signatures(model, tf_transform_output)
    model.save(fn_args.serving_model_dir,
               save_format='tf_keras', signatures=signatures)

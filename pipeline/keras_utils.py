# -*- coding:utf-8 -*-
import absl
from typing import List, Text, Dict

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


def _transform_key_name(key: Text) -> Text:
    """Add _xf to the end of the key

    Args:
        key (Text): key name

    Returns:
        Text: transformed key name
    """
    return key + '_xf'


def _make_serving_signatures(model: tf.keras.Model, tf_transform_features: tft.TFTransformOutput) -> Dict:
    """make signature

    Args:
        model (tf.keras.Model): target model
        tf_transform_features (tft.TFTransformOutput): transformed features

    Returns:
        Dict: signature info
    """
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


def _data_augment(feature_dict: Dict) -> Dict:
    """augmente image data

    Args:
        feature_dict (Dict): target features

    Returns:
        Dict: augmanted features
    """
    image_feature = feature_dict[_transform_key_name(IMG_KEY)]

    batch_size = tf.shape(image_feature)[0]
    image_feature = tf.image.random_flip_left_right(image_feature)
    image_feature = tf.image.resize_with_crop_or_pad(image_feature, 250, 250)
    image_feature = tf.image.random_crop(
        image_feature, (batch_size, IMG_SIZE, IMG_SIZE, 3))

    feature_dict[_transform_key_name(IMG_KEY)] = image_feature
    return feature_dict


def _create_dataset(
    file_pattern: List[Text],
    data_accessor: DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    is_train: bool = False,
    batch_size: int = 200
) -> tf.data.Dataset:
    """create dataset

    Args:
        file_pattern (List[Text]): List of paths or patterns of input tfrecord files.
        data_accessor (DataAccessor): DataAccessor for converting input to RecordBatch.
        tf_transform_output (tft.TFTransformOutput): A TFTransformOutput.
        is_train (bool, optional): Whether the input dataset is train split or not. Defaults to False.
        batch_size (int, optional): representing the number of consecutive elements of returned dataset to combine in a single batch. Defaults to 200.

    Returns:
        tf.data.Dataset: A dataset that contains (features, indices) tuple where features is a dictionary of Tensors, and indices is a single Tensor of label indices.
    """

    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_transform_key_name(LABEL_KEY)
        ),
        tf_transform_output.transformed_metadata.schema
    )

    if is_train:
        dataset = dataset.map(lambda x, y: (_data_augment(x), y))
    return dataset


def _build_model() -> tf.keras.Model:
    """build model

    Returns:
        tf.keras.Model: model
    """
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


def preprocessing_fn(inputs: Dict) -> Dict:
    """tf.transform's callback function

    Args:
        inputs (Dict): inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
        Dict: Map from string feature key to transformed feature operations.
    """
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
    """Train the model based on given args.

    Args:
        fn_args (TrainerFnArgs): Holds args used to train the model as name/value pairs.
    """
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

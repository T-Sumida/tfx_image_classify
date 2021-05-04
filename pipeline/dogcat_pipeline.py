# -*- coding:utf-8 -*-
import os
from absl import logging
from typing import Optional, Text, List

import tensorflow_model_analysis as tfma
from ml_metadata.proto import metadata_store_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import trainer_pb2
from tfx.proto import pusher_pb2
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.components import ImportExampleGen
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Trainer
from tfx.components import ResolverNode
from tfx.components import Evaluator
from tfx.components import Pusher
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing

PIPELINE_NAME = "dogcat_keras"

PIPELINE_ROOT = os.path.join('.', 'pipeline_output')

DATA_ROOT = os.path.join('.', 'data')

MODULE_FILE = os.path.join('pipeline', 'dogcat_keras_utils.py')

METADATA_PATH = os.path.join('.', 'metadata', PIPELINE_NAME,
                             'metadata.db')

SERVING_MODEL_DIR = os.path.join('.', 'serving_model')


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_root: Text,
    module_file: Text,
    serving_model_dir: Text,
    enable_cache: bool,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None
):

    # train testで分かれているtfrecordを指定
    input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='train', pattern='train/*'),
        example_gen_pb2.Input.Split(name='eval', pattern='test/*')
    ])
    # パイプラインにデータをロード
    example_gen = ImportExampleGen(
        input_base=data_root, input_config=input_config)

    # データの統計量を計算
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # staticsGenの統計ファイルからスキーマを生成
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

    # データに欠損などがないかを検査
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])

    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file
    )

    trainer = Trainer(
        module_file=module_file,
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=160),
        eval_args=trainer_pb2.EvalArgs(num_steps=4),
    )

    model_resolver = ResolverNode(
        instance_name='latest_blessed_model_resolver',
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing))

    # https://github.com/tensorflow/tfx/issues/3016
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(label_key='label_xf', model_type='tf_keras',
                           signature_name="serving_default")
        ],
        slicing_specs=[
            tfma.SlicingSpec(),
        ],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(
                    class_name='SparseCategoricalAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.1}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-3})))
            ])
        ])

    evaluator = Evaluator(
        examples=transform.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        evaluator,
        pusher
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )


def run_pipeline():
    my_pipeline = create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_root=DATA_ROOT,
        module_file=MODULE_FILE,
        serving_model_dir=SERVING_MODEL_DIR,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH)
    )

    BeamDagRunner().run(my_pipeline)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run_pipeline()

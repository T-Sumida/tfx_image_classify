# -*- coding:utf-8 -*-
import os
from absl import logging
from typing import Optional, Text, List

from ml_metadata.proto import metadata_store_pb2
from tfx.proto import example_gen_pb2
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.components import ImportExampleGen
from tfx.components import StatisticsGen
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

PIPELINE_NAME = "dogcat_keras"

PIPELINE_ROOT = os.path.join('.', 'pipeline_output')

DATA_ROOT = os.path.join('.', 'data')

METADATA_PATH = os.path.join('.', 'metadata', PIPELINE_NAME,
                             'metadata.db')


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_root: Text,
    enable_cache: bool,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None
):
    components = []

    input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='train', pattern='train/*'),
        example_gen_pb2.Input.Split(name='eval', pattern='test/*')
    ])
    example_gen = ImportExampleGen(
        input_base=data_root, input_config=input_config)

    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    components.append(example_gen)
    components.append(statistics_gen)

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
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH)
    )

    BeamDagRunner().run(my_pipeline)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run_pipeline()

# def _create_pipeline(
#     pipeline_name: Text,
#     pipeline_root: Text,
#     data_root: Text,
#     metadata_path: Text,
#     # enable_cache: bool,
#     metadata_connection_config: Optional[
#         metadata_store_pb2.ConnectionConfig] = None,
#     beam_pipeline_args: Optional[List[Text]] = None
# ):
#     components = []
#     return pipeline.Pipeline(
#         pipeline_name=pipeline_name,
#         pipeline_root=pipeline_root,
#         components=components,
#         # enable_cache=enable_cache,
#         metadata_connection_config=metadata_connection_config,
#         beam_pipeline_args=beam_pipeline_args,
#     )


# if __name__ == '__main__':
#     logging.set_verbosity(logging.INFO)
#     BeamDagRunner().run(
#         _create_pipeline(
#             pipeline_name=pipeline_name,
#             pipeline_root=pipeline_root,
#             data_root=data_root,
#             metadata_path=_metadata_path))

# tfx_image_classify
A sample program for image classification model using TFX.

# Overview
A pipeline project to create an image classification model using tfx.

The environment is built on wsl2, and the Cat and Dog Dataset is used as an example to build the model.

# Environment
- Windows10 home 64bit
    - RTX2080Ti
- WSL2
- pipenv

# Requirements
- tensorflow==2.3.0
- tfx==0.26.0
- Pillow


# Usage

## Directory Structure
Store the data for training and tfrecord for Evaluator under the data directory.

During training, 80% of the training data is used for training and 20% is used for evaluation.

```
tfx_image_classify
├── Pipfile
├── Pipfile.lock
├── README.md
├── data
│   ├── test
│   │   └── test.tfrecord
│   └── train
│       └── train.tfrecord
├── pipeline
│   ├── keras_utils.py
│   └── pipeline.py
└── tfrecord_generator.py
```

## Generate tfrecords
In this section, we will create a TFrecord using the sample with [Cat and Dog Dataset](https://www.kaggle.com/tongpython/cat-and-dog).

1. Download the Zip file.
2. Unzip the file so that the directory structure looks like the one below.
    ```
    tfx_image_classify
    ├── original_data
    │   ├── test_set
    │   │   └── test_set
    │   │       ├── cats
    │   │       │   ├── cat.1.jpg
    │   │       │   ├── cat.2.jpg
    │   │       │   ├── ...
    │   │       └── dogs
    │   │           ├── dog.1.jpg
    │   │           ├── dog.2.jpg
    │   │           ├── ...
    │   └── training_set
    │       └── training_set
    │           ├── cats
    │           │   ├── cat.1.jpg
    │           │   ├── cat.2.jpg
    │           │   ├── ...
    │           └── dogs
    │               ├── dog.1.jpg
    │               ├── dog.2.jpg
    │               ├── ...
    ```
3. Execute the command.
    ```
    $pipenv run generate
    ```

## Execute Pipeline
```
$pipenv run pipeline
```

The meta files and the Saved Model are output in the following directory structure.

```
.
├── Pipfile
├── Pipfile.lock
├── README.md
├── data
├── metadata
│   └── keras
│       └── metadata.db
├── pipeline
│   ├── keras_utils.py
│   └── pipeline.py
├── pipeline_output
│   ├── Evaluator
│   ├── ExampleValidator
│   ├── ImportExampleGen.test_data
│   ├── ImportExampleGen.train_data
│   ├── Pusher
│   ├── SchemaGen
│   ├── Trainer
│   └── Transform
├── serving_model
│   └── 1620117162
│       ├── assets
│       ├── saved_model.pb
│       └── variables
│           ├── variables.data-00000-of-00001
│           └── variables.index
└── tfrecord_generator.py
```


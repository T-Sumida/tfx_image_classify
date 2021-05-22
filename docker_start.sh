
# docker image を用意
docker pull tensorflow/serving

MODEL_NAME="serving_model/1620555442"
EXPORT_DIR="$(pwd)/$MODEL_NAME"
echo $EXPORT_DIR


docker run -p 8501:8501 \
--mount type=bind,source=$EXPORT_DIR,target=/models/$MODEL_NAME \
-e MODEL_NAME=$MODEL_NAME -t tensorflow/serving --model_config_file_poll_wait_seconds=60
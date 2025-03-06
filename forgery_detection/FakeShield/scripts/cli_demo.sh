# WEIGHT_PATH=./weight/fakeshield-v1-22b
WEIGHT_PATH=/data/xxp0/FakeShield/weight/fakeshield-v1-22b
IMAGE_PATH=/data/xxp0/FakeShield/playground/images/1.jpg
DTE_FDM_OUTPUT=/data/xxp0/FakeShield/playground/DTE-FDM_output.jsonl
MFLM_OUTPUT=/data/xxp0/FakeShield/playground/MFLM_output

pip install -q transformers==4.37.2  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 \
python -m llava.serve.cli \
    --model-path  ${WEIGHT_PATH}/DTE-FDM \
    --DTG-path ${WEIGHT_PATH}/DTG.pth \
    --image-path ${IMAGE_PATH} \
    --output-path ${DTE_FDM_OUTPUT}

pip install -q transformers==4.28.0  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 \
python ./MFLM/cli_demo.py \
    --version ${WEIGHT_PATH}/MFLM \
    --DTE-FDM-output ${DTE_FDM_OUTPUT} \
    --MFLM-output ${MFLM_OUTPUT}

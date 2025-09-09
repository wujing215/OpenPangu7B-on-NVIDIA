## Deployment Guide of openPangu Embedded 7B Based on [vllm-ascend](https://github.com/vllm-project/vllm-ascend)

### Deployment Environment Description

The Atlas 800T A2 (64 GB) supports the deployment of openPangu Embedded 7B (bf16) with 4 cards. The vllm-ascend community image v0.9.1-dev is used and needs to be pulled on multiple nodes.
```bash
docker pull quay.io/ascend/vllm-ascend:v0.9.1-dev
```

### Docker Boot and Inference Code

Perform the following operations on all nodes.

Run the following command to start the docker:
```bash
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:v0.9.1-dev  # Use correct image id
export NAME=vllm-ascend  # Custom docker name

# Run the container using the defined variables
# Note if you are running bridge network with docker, Please expose available ports for multiple nodes communication in advance
# To prevent device interference from other docker containers, add the argument "--privileged"
docker run --rm \
--name $NAME \
--network host \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /mnt/sfs_turbo/.cache:/root/.cache \
-it $IMAGE bash
```
If not inside the container, enter the container as the root user:
```
docker exec -itu root $NAME /bin/bash
```

Download vllm (v0.9.2) to replace the built-in vllm code of the image.
```bash
pip install --no-deps vllm==0.9.2 pybase64==1.4.1
```

Download [vllm-ascend (v0.9.2rc1)](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.9.2rc1) and replace the built-in vllm-ascend code in the image (/vllm-workspace/vllm-ascend/). For example, download [Source code (tar.gz)](https://github.com/vllm-project/vllm-ascend/archive/refs/tags/v0.9.2rc1.tar.gz) from Assets to get v0.9.2rc1.tar.gz, then extract and replace:

```bash
tar -zxvf vllm-ascend-0.9.2rc1.tar.gz -C /vllm-workspace/vllm-ascend/ --strip-components=1
export PYTHONPATH=/vllm-workspace/vllm-ascend/:${PYTHONPATH}
```

Use the Pangu model-adapted vllm-ascend code from the current repository to replace parts of the code in `/vllm-workspace/vllm-ascend/vllm_ascend/`:

```bash
yes | cp -r inference/vllm_ascend/* /vllm-workspace/vllm-ascend/vllm_ascend/
```

Replace the `tokenizer_config.json` file with the one that has added `special token`, [old file](../tokenizer_config.json) -> [new file](./vllm_ascend/tokenizer_config.json)
```bash
cp ./vllm_ascend/tokenizer_config.json ../tokenizer_config.json
```

### openPangu Embedded 7B Inference

Perform the following operations on all nodes.

Configuration:
```bash
export VLLM_USE_V1=1
# Specifying HOST=127.0.0.1 (localhost) means the server can only be accessed from the master device.
# Specifying HOST=0.0.0.0 allows the vLLM server to be accessed from other devices on the same network or even from the internet, provided proper network configuration (e.g., firewall rules, port forwarding) is in place.
HOST=xxx.xxx.xxx.xxx
PORT=8080
```

openPangu Embedded 7B running command:
```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
LOCAL_CKPT_DIR=/root/.cache/pangu_embedded_7b  # The pangu_embedded_7b bf16 weight
SERVED_MODEL_NAME=pangu_embedded_7b

vllm serve $LOCAL_CKPT_DIR \
    --served-model-name $SERVED_MODEL_NAME \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --host $HOST \
    --port $PORT \
    --max-num-seqs 32 \
    --max-model-len 32768 \
    --max-num-batched-tokens 4096 \
    --tokenizer-mode "slow" \
    --dtype bfloat16 \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.93 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
```

### Test Request

After server launched, send test request from master node or other nodes:

```bash
MASTER_NODE_IP=xxx.xxx.xxx.xxx  # server node ip
curl http://${MASTER_NODE_IP}:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$SERVED_MODEL_NAME'",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 512,
        "temperature": 0
    }'
```

# openPangu-Embedded-7B-V1.1
[中文](README.md) | English

## 1. Introduction
The openPangu-Embedded-7B-V1.1 is an efficient large language model trained from scratch based on the Ascend NPU. It contains 7 billion parameters (excluding the vocabulary embedding layer). The openPangu-Embedded-7B-V1.1 has been trained on approximately 25T tokens. The model is capable of integrating both fast and slow thinking, and can adaptively switch between two thinking mode based on assessed query complexity.


## 2. Model Architecture

|                               |   openPangu-Embedded-7B-V1.1   |
| :---------------------------: | :----------------: |
|       **Architecture**        |       Dense        |
|     **Parameters (Non-Embedding)**     |         7B         |
|     **Number of Layers**      |         34         |
|     **Hidden Dimension**      |       12800        |
|    **Attention Mechanism**    |     GQA      |
| **Number of Attention Heads** | 32 for Q，8 for KV |
|      **Vocabulary Size**      |        153k        |
|      **Context Length (Natively)**       |        32k         |
|    **Pretraining Tokens**     |        25T         |


## 3. Results

| Benchmark | Metric | Slow-thinking v1.0 | Slow-thinking v1.1 | Adaptive-switching v1.1 |
| :---: | :---: | :---: | :---: | :---: |
| **General** |  |  |  |  |
| MMLU-Pro |  Exact Match | 76.32 |   75.54    |   72.81    |
| CMMLU  |         Acc   | 75.59 |   72.94    |   72.18    |
| ArenaHard_v0.1    |   w/o style control  | 85.80 |   88.00    |   84.60    |
| C-Eval  |         Acc   | 83.05 |   84.92    |   83.33    |
| GPQA-Diamond	| Avg@4	| 70.54 |   73.23    |    73.74    |
| **Math** |  |  |            |            |
| MATH-500 | Avg@1 | 95.00 |   97.00    |   96.00    |
| AIME24 | Avg@16 | 71.57 |   79.38    |   79.02    |
| AIME25 | Avg@16 | 58.24 |   70.00    |   70.21    |
| **Coding** |  |  |            |            |
| LiveCodeBench |  Avg@2 (08/24~01/25) | 54.04 |   58.27    |   58.27    |
| MBPP+ |      Avg@2     | 76.06 |   76.46    |   75.66    |

**Note:** The system prompt is left empty, and no additional Chain-of-Thought (CoT) prompts are introduced during the evaluation. All evaluations are performed using a sequence length of 128k tokens.

In addition to accuracy, we also analyzed the model's output length on some datasets. Through data quality-driven learning strategy, adaptive-switching mode can effectively automatically switch some outputs to fast thinking on simple tasks without significantly affecting accuracy, significantly shortening the average Chain-of-Thought length. On difficult tasks, by maintaining slow thinking capabilities, the accuracy is comparable to that of a pure slow-thinking model.

|     Benchmark |   Metric       |  Slow-thinking v1.1 | Adaptive-switching v1.1 |
| :------------: | :-----------------: |  :--------: | :--------: |
|  **General**  |                    |            |            |
|     CMMLU      |         Acc        |   72.94    |   72.18    |
|        |   Length     |    2574    |   1338    |
|     C-Eval     |         Acc         |     84.92    |   83.33    |
|        |   Length     |  2484    |   1723    |
|  **Math**  |               |            |            |
|     AIME24     |       Avg@16        |     79.38    |   79.02    |
|        |   Length     |    48229    |   49656   |
|  **Coding**  |                |            |            |
| LiveCodeBench  | Avg@2 (08/24~01/25) |    58.27    |   58.27    |
|        |   Length     | 58140    |   59307    |

## 4. Deployment

### 4.1 Environment

##### Hardware Requirements

Atlas 800T A2 (64GB), please refer to [[Atlas 800T A2](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.2.RC1.alpha003&driver=Ascend+HDK+25.0.RC1)] for obtaining the driver and firmware installation packages.

#### System Requirements & Dependencies

- System: Linux (OpenEuler ≥ 24.03 recommended)
- CANN==8.1.RC1: [[CANN Install]](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)
- python==3.10
- torch==2.1.0
- torch-npu==2.1.0.post12
- transformers==4.53.2

The above software environment has been verified, and theoretically supports newer versions. For any questions, please submit an issue.

### 4.2 Integrity Check

Please refer to the following methods to verify the integrity of the downloaded content. The hash values are stored in the `checklist.chk` file.

```
#!/usr/bin/env bash
ARCH=$(uname -m)
MODEL_PATH="${TARGET_FOLDER}/${MODEL_FOLDER_PATH}"
cd "$MODEL_PATH" || exit 1
if [ "$ARCH" = "arm64" ]; then
    sha256sum checklist.chk
else
    sha256sum -c checklist.chk
fi
```

### 4.3 Inference Examples

The following provides a simple inference example of openPangu-Embedded-7B-V1.1 based on the `transformers` framework: 
>Please modify generate.py and add the model path before running.
```bash
cd inference
python generate.py
```

The openPangu-Embedded-7B-V1.1 model is in slow thinking mode by default, and can be switched to adaptive/fast thinking mode by the following means:
- In the code example `generate.py`, the definition of the `auto_thinking_prompt` and `no_thinking_prompt` variables demonstrates the specific implementation for switching to adaptive/fast thinking mode: by appending the `/auto_think` or `/no_think` tag at the end of user input, the current turn can be switched to fast thinking mode. In this mode, `thinking_content` will be an empty value.

### 4.4 Using Inference Framework
vllm_ascend：[[vllm_ascend_for_openpangu_embedded_7b]](inference/vllm_ascend_for_openpangu_embedded_7b.md)




## 5. Model License

Unless otherwise noted, openPangu-Embedded-7B-V1.1 model is licensed under the terms and conditions of OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0, which is intended to be used permissively and enable the further development of artificial intelligence technologies. Please refer to the [LICENSE](LICENSE) file located in the root directory of the model repository for details.

## 6. Disclaimer
Due to the technical limitations inherent in the technology on which the openPangu-Embedded-7B-V1.1 (“Model”) relies and the fact that the artificial intelligence generated content is automatically produced by Model, Huawei cannot make any guarantees regarding the following matters:
- The output of this Model is automatically generated via AI algorithms, it does not rule out the possibility that some of the information may be flawed, unreasonable, or cause discomfort, and the generated content does not represent Huawei's attitude or standpoint;
- There is no guarantee that this Model is 100% accurate, reliable, functional, timely, secure and safety, error-free, uninterrupted, continuously stable, or free of any faults;
- The output of this Model does not constitute any advices or decisions for you, and it does not guarantee the authenticity, completeness, accuracy, timeliness, legality, functionality, or practicality of the generated content. The generated content cannot replace professionals in medical, legal, and other fields in answering your questions. The generated content is for your reference only and does not represent any attitude, standpoint, or position of Huawei. You need to make independent judgments based on your actual situation, and Huawei does not assume any responsibilities.


## 7. Contact Us
If you have any comments or suggestions, please submit an issue or contact openPangu@huawei.com.
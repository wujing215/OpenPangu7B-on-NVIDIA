# 开源盘古 Embedded-7B-V1.1

中文 | [English](README_EN.md)

## 1. 简介

openPangu-Embedded-7B-V1.1 是基于昇腾 NPU 从零训练的高效大语言模型，参数量为 7B（不含词表Embedding）。openPangu-Embedded-7B-V1.1 训练了约 25T tokens，具备快慢思考融合与自适应切换能力。

## 2. 模型架构

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

## 3. 测评结果

|     测评集     |      测评指标       | 慢思考v1.0 | 慢思考v1.1 | 自适应v1.1 |
| :------------: | :-----------------: | :--------: | :--------: | :--------: |
|  **通用能力**  |                     |            |            |            |
|    MMLU-Pro    |     Exact Match     |   76.32    |   75.54    |   72.81    |
|     CMMLU      |         Acc         |   75.59    |   72.94    |   72.18    |
| ArenaHard_v0.1 |  w/o style control  |   85.80    |   88.00    |   84.60    |
|     C-Eval     |         Acc         |   83.05    |   84.92    |   83.33    |
|  GPQA-Diamond  |        Avg@4        |   70.54    |   73.23    |    73.74    |
|  **数学能力**  |                     |            |            |            |
|    MATH-500    |        Avg@1        |   95.00    |   97.00    |   96.00    |
|     AIME24     |       Avg@16        |   71.57    |   79.38    |   79.02    |
|     AIME25     |       Avg@16        |   58.24    |   70.00    |   70.21    |
|  **代码能力**  |                     |            |            |            |
| LiveCodeBench  | Avg@2 (08/24~01/25) |   54.04    |   58.27    |   58.27    |
|     MBPP+      |        Avg@2        |   76.06    |   76.46    |   75.66    |

**注：** 评测过程中system prompt 为空，且不添加任何额外的思维链（CoT）提示。评测采用 128k 的序列长度进行。

除精度外，我们还在部分数据集上统计了模型的输出长度，通过数据质量驱动的学习策略，自适应快慢思考可以在基本不影响精度地前提下，有效地在简单任务上自动切换部分输出为快思考，大幅缩短平均输出思维链长度（Length）；在难任务通过保持慢思考能力，精度持平纯慢思考模型。

|     测评集     |      测评指标       |  慢思考v1.1 | 自适应v1.1 |
| :------------: | :-----------------: |  :--------: | :--------: |
|  **通用能力**  |                    |            |            |
|     CMMLU      |         Acc        |   72.94    |   72.18    |
|        |   Length     |    2574    |   1338    |
|     C-Eval     |         Acc         |     84.92    |   83.33    |
|        |   Length     |  2484    |   1723    |
|  **数学能力**  |               |            |            |
|     AIME24     |       Avg@16        |     79.38    |   79.02    |
|        |   Length     |    48229    |   49656   |
|  **代码能力**  |                |            |            |
| LiveCodeBench  | Avg@2 (08/24~01/25) |    58.27    |   58.27    |
|        |   Length     | 58140    |   59307    |

## 4. 部署和使用

### 4.1 环境准备

##### 硬件规格

Atlas 800T A2 (64GB)，驱动与固件安装包获取请参照 [[Atlas 800T A2](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.2.RC1.alpha003&driver=Ascend+HDK+25.0.RC1)]。

##### 软件环境

- 操作系统：Linux（推荐 openEuler>=24.03）
- CANN==8.1.RC1，安装准备及流程请参照 [[CANN Install]](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)
- python==3.10
- torch==2.1.0
- torch-npu==2.1.0.post12
- transformers==4.53.2

以上软件配套经过验证，理论可以支持更高版本，如有疑问，可以提交 issue。

### 4.2 权重完整性校验

请参考以下方法对下载内容进行完整性校验，hash 值存储在 checklist.chk 文件中。

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

### 4.3 推理样例

下述内容提供 openPangu-Embedded-7B-V1.1 在 `transformers` 框架上进行推理的一个简单示例：

> 运行前请修改 generate.py，添加模型路径。

```bash
cd inference
python generate.py
```

openPangu-Embedded-7B-V1.1 模型默认为慢思考模式，可以通过以下手段切换至快慢自适应切换/快思考模式：

- 在代码实例`generate.py`中，`auto_thinking_prompt`与`no_thinking_prompt`变量的定义展示了切换至快慢自适应或快思考模式的具体实现：通过在用户输入末尾添加`/auto_think`或`/no_think`标记，可将当前轮次切换至快慢自适应切换/快思考模式。

### 4.4 使用推理框架

vllm_ascend：参考[[vllm_ascend_for_openpangu_embedded_7b.zh]](inference/vllm_ascend_for_openpangu_embedded_7b.zh.md)

## 5. 模型许可证

除文件中对开源许可证另有约定外，openPangu-Embedded-7B-V1.1 模型根据 OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0 授权，旨在允许使用并促进人工智能技术的进一步发展。有关详细信息，请参阅模型存储库根目录中的 [LICENSE](LICENSE) 文件。

## 6. 免责声明

由于 openPangu-Embedded-7B-V1.1（“模型”）所依赖的技术固有的技术限制，以及人工智能生成的内容是由盘古自动生成的，华为无法对以下事项做出任何保证：

- 尽管该模型的输出由 AI 算法生成，但不能排除某些信息可能存在缺陷、不合理或引起不适的可能性，生成的内容不代表华为的态度或立场；
- 无法保证该模型 100% 准确、可靠、功能齐全、及时、安全、无错误、不间断、持续稳定或无任何故障；
- 该模型的输出内容不构成任何建议或决策，也不保证生成的内容的真实性、完整性、准确性、及时性、合法性、功能性或实用性。生成的内容不能替代医疗、法律等领域的专业人士回答您的问题。生成的内容仅供参考，不代表华为的任何态度、立场或观点。您需要根据实际情况做出独立判断，华为不承担任何责任。

## 7. 反馈

如果有任何意见和建议，请提交issue或联系 openPangu@huawei.com。

# llm-post-training

项目一：在 Qwen2.5-0.5B 上把后训练整条链路自己走一遍，用 GSM8K（小学数学应用题）量化每一步的效果。

计划：Base → SFT（手写 LoRA）→ DPO（自己造偏好数据、手写 loss）→ GRPO（手写优势和 loss，加一项改进对照）。评测脚本自己写。

## 对照表（GSM8K test 1319 题，贪心解码）

| 模型 | 准确率 | 平均回答长度 | 说明 |
|---|---|---|---|
| Qwen2.5-0.5B（Base） | 待测 | | 起点 |
| + SFT（LoRA） | 待测 | | |
| + DPO | 待测 | | |
| + GRPO | 待测 | | |
| Qwen2.5-0.5B-Instruct | 待测 | | 官方后训练版，作参照 |

## 环境

- 1 × RTX 3090；Python 3.10.22，torch 2.14.1+cu130，transformers 5.20.2
- 权重从 ModelScope 下载（`Qwen/Qwen2.5-0.5B`、`Qwen/Qwen2.5-0.5B-Instruct`），放在仓库外，不进 git

## 文件

### `01_hello_qwen.py`：看清一个真正的大模型

- **分词**：词表 151665 个 token（BPE）。数字一位一个 token（`48` → `4`、`8`），生僻词拆成几块（`Natalia` → `N`、`atal`、`ia`）。
- **对话模板**：`apply_chat_template` 用 `<|im_start|>角色` … `<|im_end|>` 包住每段话，`add_generation_prompt=True` 在末尾补上 `<|im_start|>assistant`。一条 system 加一条 user 拼完是 25 个 token；结束符 `<|im_end|>` 的编号是 151645。
- **模型**：参数量 494,032,768（embedding 和 lm_head 共用一张表）。和自己写的 nanoGPT 比，换了四样：RoPE 代替位置表，RMSNorm 代替 LayerNorm，GQA（14 个 Q 头共用 2 组 K/V），SwiGLU 的 MLP。
- **生成**：官方 `generation_config` 默认抽样（temperature 0.7、top_k 20、top_p 0.8、repetition_penalty 1.1）。评测要可复现，改成贪心（`do_sample=False`、`repetition_penalty=1.0`）。问"1+1等于几？"，回答 `1+1等于2。`（7 个新 token）。
- **手写贪心循环**：每轮把整段喂进去，取最后一个位置的 argmax 接到后面，写出 `<|im_end|>` 就停。和 `generate` 生成的 32 个编号逐个相同。
- **故意改错**：循环只转 3 轮，形状 (1, 28) 对 (1, 32)，assert 挂；参数量写成 494032769，挂在第 26 行，打印出实际值 494032768。
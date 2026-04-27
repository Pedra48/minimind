# Tokenizer

## Tokenizer 的训练

Tokenizer训练流程如下：预分词 --> 使用BPE方法训练tokenizer --> 收尾工作

总的来说，minimind中的tokenizer的预训练使用BPE的方法，这种方法通过出现的频率进行分词，直到达到预先设定的词表大小。`tokenizer = Tokenizer(models.BPE())`代码指定了这里的分词模型算法为BPE(Byte Pair Encoding)。其他别的算法包括Unigram,WordLevel,WordPiece等。

### 预分词
预分词的目标是将用于训练的文本集按照设定好的粒度进行切分。在minimind中，考虑采用ByteLevel进行预分词。`tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)`。

预分词的方法有很多，以下针对其中的一部分进行简单介绍。

| 预分词方法 | 核心逻辑 | 优缺点 / 特点 | 适用模型 |
| :--- | :--- | :--- | :--- |
| **Whitespace** | 仅根据空格、制表符、换行符进行切分。 | **优点：** 极其简单。<br>**缺点：** 会把标点和单词连在一起（如 "Hi!"），导致词表冗余。 | 早期 NLP 模型 |
| **Punctuation** | 在所有标点符号的前后强制增加切分边界。 | **优点：** 保证标点独立成词。<br>**缺点：** 无法处理缩写（如 "can't" 变成 "can", "'", "t"）。 | BERT, RoBERTa |
| **Digits** | 将连续的数字拆分为单个数字（或按位数拆分）。 | **优点：** 防止模型死记硬背大数字，提高对数值计算的敏感度。 | 绝大多数现代 LLM |
| **Metaspace** | 将空格替换为一个特殊字符（如 ` ` 或 `_`），再按空格切分。 | **优点：** 能够完全无损地还原原始文本（包括空格的精确数量）。 | SentencePiece, Llama, ALBERT |
| **Regex** | 使用复杂的正则表达式（基于字符类型）进行切分。 | **优点：** 极其灵活，可同时处理中英文混排、数字和缩写。<br>**缺点：** 正则表达式的设计非常复杂。 | GPT-4, Llama 3, Tiktoken |
| **Byte-Level** | **(MiniMind 采用)** 将 UTF-8 字节映射为可见的 Unicode 字符。 | **优点：** 彻底解决 OOV（词表外）问题，任何字符都能表示，不会出现 `[UNK]`。 | GPT-2, MiniMind, Llama 3 |

这里的ByteLevel对语料进行逐字节的切分。例如对于语料"Hi" 会切分为['H','I']，而如果是汉字这样的，如果一个字占两个字节，就会把每个字切分为两个字节。
具体来说，ByteLevel考虑一个字节映射表，举例说，对于一个字"学"，把它的三个字节分别映射到一个Unicode字符上，形成逐字节的拆分。

### 训练Tokenizer

在进行正式地对tokenizer进行训练之前，还需要一个工程实现上的小技巧：在设定完特殊token后，还要预留一定数量的token位置作为buffer_token，这是为了让不同版本的模型可以共用同一份基础词表，只需要在buffer的位置上填充各自模型需要的特殊token即可。

随后进行正式训练。
首先，初始化训练器：
```
trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, # 词表大小
        show_progress=True,    # 显示训练进度
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), # 初识字母表
        special_tokens=all_special_tokens # 必须保留的特殊token
    )
```
随后，使用迭代的方式训练tokenizer。具体来说，在每个iteration中，找到语料中的高频token对，把它们合并成一个token，然后再投入到下一轮iteration中，直到词表大小达到预先设定好的vocab_size。

最后，我们需要通过decoder把分词后的token序列还原回文本，具体而言，需要合并子词、移除添加过的前缀，把unicode码重新解码为人类可以阅读的文字。

### 收尾工作

这部分主要是进行一些工程上的东西。
1. 需要整理是否为特殊token
2. 对所有的特殊token添加详细信息。
3. 保存为json格式
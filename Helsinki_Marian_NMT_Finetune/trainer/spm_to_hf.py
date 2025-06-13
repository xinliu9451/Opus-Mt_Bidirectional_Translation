import sentencepiece as spm
from transformers import T5Tokenizer

# 使用 T5Tokenizer 来加载 SentencePiece 模型
# T5Tokenizer 原生支持 SentencePiece 模型
tokenizer = T5Tokenizer(
    vocab_file="./tokenizer/zh_en_mix_65001.model",
    extra_ids=0,
    model_max_length=512
)

# 添加你训练时指定的 special tokens
tokenizer.add_special_tokens({
    "pad_token": "<pad>",         # 你训练时 pad_id=0，但没设置具体字符串；可设为 "<pad>"
    "unk_token": "<unk>",         # unk_id=1
    "eos_token": "</s>",          # eos_id=3，对应你控制符中的 </s>
    "additional_special_tokens": [
        "<cmd>", ">>eng<<", ">>zho<<"  # 你的 control symbols
    ]
})

# 测试 tokenizer 是否工作正常
test_text = "Hello world 你好世界"
tokens = tokenizer.encode(test_text)
print(f"测试文本: {test_text}")
print(f"编码结果: {tokens}")
print(f"解码结果: {tokenizer.decode(tokens)}")

# 保存为 HF 格式（可直接被 tokenizer.from_pretrained() 使用）
tokenizer.save_pretrained("zh_en_spm_hf_tokenizer")
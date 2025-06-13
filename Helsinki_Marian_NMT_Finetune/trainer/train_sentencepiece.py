import sentencepiece as spm

# # 训练模型
spm.SentencePieceTrainer.Train(
    input='./data/train_sentencepiece_texts.txt',
    model_prefix='zh_en_mix_65001',
    vocab_size=65001,
    character_coverage=1.0,
    model_type='bpe',

    # 这几个 token 的 ID 必须严格对齐原模型配置
    pad_id=65000,
    unk_id=1,
    bos_id=-1,         # Marian 未使用 <s>（通常由 decoder_start_token_id 控制）
    eos_id=0,

    user_defined_symbols=["</s>", "<cmd>", ">>eng<<", ">>zho<<"],

    input_sentence_size=20000000,
    shuffle_input_sentence=True,
    num_threads=64
)



# # 验证模型
# sp = spm.SentencePieceProcessor()
# sp.load("/data/chenkj/Translate_finetune/Helsinki_Marian_NMT/zh_en_mix_65001.model")

# text = ">>eng<< 我爱你，who are you, 我喜欢dog。你喜欢cat嘛？"
# pieces = sp.encode(text, out_type=str)
# ids = sp.encode(text, out_type=int)

# print("Pieces:", pieces)
# print("IDs:", ids)

# # 解码
# decoded = sp.decode(ids)
# print("Decoded:", decoded)


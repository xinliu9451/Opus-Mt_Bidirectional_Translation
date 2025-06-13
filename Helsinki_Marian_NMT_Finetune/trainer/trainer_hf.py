import torch
import evaluate
import zhconv
from datasets import load_dataset, Dataset, IterableDataset
from transformers import (
    AutoTokenizer, MarianMTModel,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback, T5Tokenizer
)
import pandas as pd
import sacrebleu
import os
from torch.utils.data import IterableDataset as TorchIterableDataset
import random

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model/Helsinki-opus-mt-zh-en")

# 加载模型
model = MarianMTModel.from_pretrained("./model/Helsinki-opus-mt-zh-en")

# 设置 token ID
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

class BilingualIterableDataset(TorchIterableDataset):
    """自定义的可迭代数据集，支持双向翻译和流式处理"""
    
    def __init__(self, file_path, tokenizer, max_length=128, is_train=True, train_ratio=0.999):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.train_ratio = train_ratio
        
        # 计算数据集大小（只读一遍文件）
        self.total_lines = self._count_lines()
        self.train_size = int(self.total_lines * self.train_ratio)
        
    def _count_lines(self):
        """计算文件行数"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # 跳过表头
            next(f)
            count = sum(1 for _ in f)
        return count
    
    def _read_data_stream(self):
        """流式读取数据"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # 跳过表头
            header = next(f).strip().split('\t')
            chinese_idx = header.index('Chinese')
            english_idx = header.index('English')
            
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) <= max(chinese_idx, english_idx):
                    continue
                    
                chinese = parts[chinese_idx].strip()
                english = parts[english_idx].strip()
                
                # 过滤空数据
                if not chinese or not english:
                    continue
                
                # 根据训练/验证比例决定是否包含这条数据
                if self.is_train and line_idx >= self.train_size:
                    continue
                elif not self.is_train and line_idx < self.train_size:
                    continue
                
                yield chinese, english, line_idx
    
    def _preprocess_zh2en(self, chinese, english):
        """中译英预处理"""
        input_text = ">>eng<< " + chinese  # 双向翻译需要加一个目标语言标识
        
        input_encoded = self.tokenizer(
            input_text, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length"
        )
        
        output_encoded = self.tokenizer(
            english, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length"
        )
        
        return {
            "input_ids": input_encoded["input_ids"],
            "attention_mask": input_encoded["attention_mask"],
            "decoder_input_ids": output_encoded["input_ids"],
            "decoder_attention_mask": output_encoded["attention_mask"],
            "labels": output_encoded["input_ids"].copy(),
        }
    
    def _preprocess_en2zh(self, chinese, english):
        """英译中预处理"""
        input_text = ">>zho<< " + english
        
        input_encoded = self.tokenizer(
            input_text, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length"
        )
        
        output_encoded = self.tokenizer(
            chinese, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length"
        )
        
        return {
            "input_ids": input_encoded["input_ids"],
            "attention_mask": input_encoded["attention_mask"],
            "decoder_input_ids": output_encoded["input_ids"],
            "decoder_attention_mask": output_encoded["attention_mask"],
            "labels": output_encoded["input_ids"].copy(),
        }
    
    def __iter__(self):
        """迭代器，返回处理后的数据"""
        for chinese, english, line_idx in self._read_data_stream():
            # 随机决定是中译英还是英译中（实现双向翻译）
            if random.random() < 0.5:
                # 中译英
                yield self._preprocess_zh2en(chinese, english)
            else:
                # 英译中
                yield self._preprocess_en2zh(chinese, english)
    
    def __len__(self):
        """返回数据集大小"""
        if self.is_train:
            return self.train_size * 2  # 双向翻译，所以是2倍
        else:
            return (self.total_lines - self.train_size) * 2

# 创建训练和验证数据集
train_dataset = BilingualIterableDataset(
    "./data/train_data.tsv", 
    tokenizer, 
    is_train=True, 
    train_ratio=0.9999
)

eval_dataset = BilingualIterableDataset(
    "./data/train_data.tsv", 
    tokenizer, 
    is_train=False, 
    train_ratio=0.9999
)

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    bleu = sacrebleu.corpus_bleu(pred_str, [label_str])

    # 保存到本地文件
    save_dir = "./eval_logs"
    os.makedirs(save_dir, exist_ok=True)
    eval_id = f"step_{trainer.state.global_step}" if 'trainer' in globals() and hasattr(trainer, "state") else "eval"
    output_file = os.path.join(save_dir, f"pred_vs_ref_{eval_id}.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, (pred, ref) in enumerate(zip(pred_str, label_str)):
            f.write(f"Sample {i + 1}:\n")
            f.write(f"Prediction: {pred}\n")
            f.write(f"Reference : {ref}\n")
            f.write("=" * 50 + "\n")

    return {
        "bleu": bleu.score
    }

# 训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir='./model/marian-zh-en-bidirectional',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=1000,
    save_steps=5000,
    eval_steps=5000,
    eval_strategy="steps",
    predict_with_generate=True,
    save_total_limit=10,
    report_to="tensorboard",
    logging_dir='./logs',
    dataloader_num_workers=2,  # 启用多进程数据加载
    dataloader_pin_memory=True,  # 启用内存固定
    gradient_accumulation_steps=2,  # 梯度累积，减少内存使用

    # 添加这几项：自动保存BLEU最高的模型
    load_best_model_at_end=True,
    metric_for_best_model="bleu",     # 你 compute_metrics 中需要返回 "bleu"
    greater_is_better=True,
)

# 构建 early stopping callback，设置 patience
early_stopping = EarlyStoppingCallback(early_stopping_patience=6)

# 构建 Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  # 使用 processing_class 代替 tokenizer
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],  # 加入 callback
)

# 开始训练
trainer.train(resume_from_checkpoint=False)

# 保存模型和 tokenizer
model.save_pretrained("./model/marian-zh-en-bidirectional")
tokenizer.save_pretrained("./model/marian-zh-en-bidirectional")

print("训练完成！模型已保存。")
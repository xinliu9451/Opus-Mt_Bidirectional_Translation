import sentencepiece as spm
from typing import List, Dict, Any, Union
import os

class SentencePieceTokenizer:
    """SentencePiece的包装类，兼容Hugging Face tokenizer接口"""
    
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        
        # 保存原始模型路径，用于save_pretrained
        self.model_path = model_path
        
        # 定义特殊token ID
        self.pad_token_id = 65000  # 这是为了保持和 Helsinki-NLP/opus-mt-zh-en 一致
        self.unk_token_id = 1
        self.bos_token_id = -1
        self.eos_token_id = 0
        
        # 定义特殊token
        self.pad_token = self.sp.IdToPiece(self.pad_token_id)
        self.unk_token = self.sp.IdToPiece(self.unk_token_id)
        self.bos_token = None
        self.eos_token = "</s>"
        
        # 添加vocab_size属性（HuggingFace兼容）
        self.vocab_size = self.sp.GetPieceSize()
        
        # 添加model_max_length属性
        self.model_max_length = 512
        
        # 添加padding_side属性（DataCollator需要）
        self.padding_side = "right"  # 默认在右侧填充
        
        # 添加其他可能需要的属性
        self.truncation_side = "right"
        self.is_fast = False  # 表示这不是一个fast tokenizer
        
    def __call__(self, text: str, max_length: int = 128, truncation: bool = True, 
                 padding: str = "max_length", return_tensors: str = None) -> Dict[str, Any]:
        """模拟HF tokenizer的__call__方法"""
        if isinstance(text, list):
            # 批量处理
            input_ids = []
            attention_mask = []
            for t in text:
                ids, mask = self._encode_single(t, max_length, truncation, padding)
                input_ids.append(ids)
                attention_mask.append(mask)
        else:
            # 单个文本
            input_ids, attention_mask = self._encode_single(text, max_length, truncation, padding)
            input_ids = [input_ids]
            attention_mask = [attention_mask]
        
        # 如果只有一个样本且不需要批处理维度，去掉外层列表
        if not isinstance(text, list) and len(input_ids) == 1:
            result = {
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0]
            }
        else:
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        
        return result
    
    def _encode_single(self, text: str, max_length: int, truncation: bool, padding: str):
        """编码单个文本"""
        # 编码文本
        tokens = self.sp.EncodeAsIds(text)
        
        # 添加结束符
        tokens = tokens + [self.eos_token_id]
        
        # 截断
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.eos_token_id]
        
        # 创建attention mask
        attention_mask = [1] * len(tokens)
        
        # 填充
        if padding == "max_length":
            padding_length = max_length - len(tokens)
            if padding_length > 0:
                tokens = tokens + [self.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
        
        return tokens, attention_mask
    
    def batch_decode(self, sequences: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """批量解码"""
        texts = []
        for seq in sequences:
            # 转换为列表（如果是tensor）
            if hasattr(seq, 'tolist'):
                seq = seq.tolist()
            
            # 移除padding和特殊token
            if skip_special_tokens:
                seq = [token_id for token_id in seq if token_id not in [self.pad_token_id, self.eos_token_id, self.bos_token_id, -100]]
            
            # 解码
            text = self.sp.DecodeIds(seq)
            texts.append(text)
        
        return texts
    
    def pad(self, encoded_inputs, padding=True, max_length=None, return_tensors=None, **kwargs):
        """模拟HF tokenizer的pad方法，用于DataCollator"""
        
        # 如果是单个输入，转换为列表
        if isinstance(encoded_inputs, dict) and "input_ids" in encoded_inputs:
            if not isinstance(encoded_inputs["input_ids"][0], list):
                # 单个样本，已经是正确格式
                if return_tensors == "pt":
                    for key in encoded_inputs:
                        encoded_inputs[key] = torch.tensor(encoded_inputs[key])
                return encoded_inputs
            else:
                # 多个样本在一个dict中
                batch_size = len(encoded_inputs["input_ids"])
                keys = list(encoded_inputs.keys())
                batch = [{key: encoded_inputs[key][i] for key in keys} for i in range(batch_size)]
                encoded_inputs = batch
        
        # 处理批量输入
        if not isinstance(encoded_inputs, list):
            encoded_inputs = [encoded_inputs]
        
        # 获取所有序列的长度
        if max_length is None:
            # 找到批次中最长的序列
            max_length = max(len(inputs.get("input_ids", [])) for inputs in encoded_inputs)
        
        # 填充每个序列
        padded_inputs = {"input_ids": [], "attention_mask": []}
        if "decoder_input_ids" in encoded_inputs[0]:
            padded_inputs["decoder_input_ids"] = []
        if "decoder_attention_mask" in encoded_inputs[0]:
            padded_inputs["decoder_attention_mask"] = []
        if "labels" in encoded_inputs[0]:
            padded_inputs["labels"] = []
        
        for inputs in encoded_inputs:
            # 填充input_ids
            input_ids = inputs.get("input_ids", [])
            padding_length = max_length - len(input_ids)
            padded_ids = input_ids + [self.pad_token_id] * padding_length
            padded_inputs["input_ids"].append(padded_ids)
            
            # 填充attention_mask
            attention_mask = inputs.get("attention_mask", [1] * len(input_ids))
            padded_mask = attention_mask + [0] * padding_length
            padded_inputs["attention_mask"].append(padded_mask)
            
            # 如果有decoder_input_ids，也需要填充
            if "decoder_input_ids" in inputs:
                decoder_ids = inputs["decoder_input_ids"]
                decoder_padding_length = max_length - len(decoder_ids)
                padded_decoder_ids = decoder_ids + [self.pad_token_id] * decoder_padding_length
                padded_inputs["decoder_input_ids"].append(padded_decoder_ids)
            
            # 填充decoder_attention_mask
            if "decoder_attention_mask" in inputs:
                decoder_mask = inputs["decoder_attention_mask"]
                decoder_padding_length = max_length - len(decoder_mask)
                padded_decoder_mask = decoder_mask + [0] * decoder_padding_length
                padded_inputs["decoder_attention_mask"].append(padded_decoder_mask)
            
            # 填充labels（使用-100作为padding值，这是PyTorch的忽略索引）
            if "labels" in inputs:
                labels = inputs["labels"]
                labels_padding_length = max_length - len(labels)
                padded_labels = labels + [-100] * labels_padding_length
                padded_inputs["labels"].append(padded_labels)
        
        # 转换为张量（如果需要）
        if return_tensors == "pt":
            try:
                import torch
                for key in padded_inputs:
                    padded_inputs[key] = torch.tensor(padded_inputs[key])
            except ImportError:
                print("警告: 无法导入torch，返回列表格式而非张量")
        
        return padded_inputs
    
    def save_pretrained(self, save_directory: str):
        """保存tokenizer（复制原始模型文件）"""
        os.makedirs(save_directory, exist_ok=True)
        import shutil
        
        # 使用初始化时的模型路径
        model_path = self.model_path
        
        # 推断vocab文件路径（将.model替换为.vocab）
        vocab_path = model_path.replace('.model', '.vocab')
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在: {model_path}")
            return
        
        if not os.path.exists(vocab_path):
            print(f"警告: 词汇表文件不存在: {vocab_path}")
            # 如果vocab文件不存在，只复制model文件
            shutil.copy2(model_path, 
                         os.path.join(save_directory, "sentencepiece.model"))
            print(f"已保存模型文件到: {save_directory}")
            return
        
        # 复制模型和词汇表文件
        shutil.copy2(model_path, 
                     os.path.join(save_directory, "sentencepiece.model"))
        shutil.copy2(vocab_path, 
                     os.path.join(save_directory, "sentencepiece.vocab"))
        
        print(f"已保存tokenizer文件到: {save_directory}")
        print(f"  - sentencepiece.model (来源: {model_path})")
        print(f"  - sentencepiece.vocab (来源: {vocab_path})") 
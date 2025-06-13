from collections import Counter
import math
import re

def bleu_score(reference, candidate, weights=[0.25, 0.25, 0.25, 0.25]):
    """
    计算BLEU分数
    reference: 参考句子(字符串)
    candidate: 候选句子(字符串) 
    weights: n-gram权重
    """
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    # 计算brevity penalty
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    
    if cand_len == 0:
        return 0.0
    
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len/cand_len)
    
    # 计算n-gram precision
    precisions = []
    for n in range(1, len(weights) + 1):
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        cand_ngrams = Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens)-n+1)])
        
        overlap = sum(min(ref_ngrams[ng], cand_ngrams[ng]) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())
        
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(overlap / total)
    
    # 计算几何平均
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_precisions = [weights[i] * math.log(precisions[i]) for i in range(len(precisions))]
    geo_mean = math.exp(sum(log_precisions))
    
    return bp * geo_mean

def chrf_score(reference, candidate, beta=2, n_char=6, n_word=2):
    """
    计算chrF分数
    reference: 参考句子
    candidate: 候选句子
    beta: F-score中的beta参数
    n_char: 字符n-gram的最大长度
    n_word: 词n-gram的最大长度
    """
    def get_char_ngrams(text, n):
        """获取字符n-gram"""
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i+n])
        return ngrams
    
    def get_word_ngrams(text, n):
        """获取词n-gram"""
        words = text.split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return ngrams
    
    # 去除空格进行字符级别计算
    ref_chars = reference.replace(' ', '')
    cand_chars = candidate.replace(' ', '')
    
    char_precision_sum = 0
    char_recall_sum = 0
    
    # 计算字符n-gram的precision和recall
    for n in range(1, n_char + 1):
        ref_ngrams = Counter(get_char_ngrams(ref_chars, n))
        cand_ngrams = Counter(get_char_ngrams(cand_chars, n))
        
        if not cand_ngrams:
            continue
            
        overlap = sum(min(ref_ngrams[ng], cand_ngrams[ng]) for ng in cand_ngrams)
        
        precision = overlap / sum(cand_ngrams.values()) if sum(cand_ngrams.values()) > 0 else 0
        recall = overlap / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0
        
        char_precision_sum += precision
        char_recall_sum += recall
    
    # 计算词n-gram的precision和recall
    word_precision_sum = 0
    word_recall_sum = 0
    
    for n in range(1, n_word + 1):
        ref_ngrams = Counter(get_word_ngrams(reference, n))
        cand_ngrams = Counter(get_word_ngrams(candidate, n))
        
        if not cand_ngrams:
            continue
            
        overlap = sum(min(ref_ngrams[ng], cand_ngrams[ng]) for ng in cand_ngrams)
        
        precision = overlap / sum(cand_ngrams.values()) if sum(cand_ngrams.values()) > 0 else 0
        recall = overlap / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0
        
        word_precision_sum += precision
        word_recall_sum += recall
    
    # 平均precision和recall
    total_precision = (char_precision_sum + word_precision_sum) / (n_char + n_word)
    total_recall = (char_recall_sum + word_recall_sum) / (n_char + n_word)
    
    # 计算F-score
    if total_precision + total_recall == 0:
        return 0.0
    
    chrf = (1 + beta**2) * total_precision * total_recall / (beta**2 * total_precision + total_recall)
    return chrf

def compute_corpus_metrics(predictions, references):
    """
    计算语料库级别的BLEU和chrF分数
    """
    if len(predictions) != len(references):
        raise ValueError("预测和参考句子数量不匹配")
    
    # 计算每个句子的分数
    bleu_scores = []
    chrf_scores = []
    
    for pred, ref in zip(predictions, references):
        bleu_scores.append(bleu_score(ref, pred))
        chrf_scores.append(chrf_score(ref, pred))
    
    # 计算平均分数
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_chrf = sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0
    
    return {
        'bleu': avg_bleu,
        'chrf': avg_chrf,
        'bleu_scores': bleu_scores,
        'chrf_scores': chrf_scores
    }

# Opus-Mt_Bidirectional_Translation
Fine tune the Helsinki NLP Opus MT model to achieve bidirectional translation of a single model

Instructions for use in myct.ipynb. Additionally, training the bidirectional translation model can refer to:https://blog.csdn.net/m0_62603533/article/details/148342089


## Updated  Helsinki_Marian_NMT_Finetune

data：They are the data for training the translation model and the tokenizer, respectively. As it is a bidirectional translation, the data for training the tokenizer is a mixture of Chinese and English.

model：You need to download the Helsinki-NLP/opus-mt-zh-en model from HF.

tokenizer：This is the tokenizer I trained using 20 million mixed Chinese English data.

trainer：This directory contains the main training code. sentencepiece_tokenizer.py is a self-made sentencepiece class designed to adapt to the training code of the transformer library.  spm_to_hf. py is the code that converts the sentence piece tokenizer into HF format. train_sentencepiece.py is the code for training a tokenizer.  trainr_hf.py is the training code that uses an HF tokenizer for encoding and decoding.  trainer_sentencepiece.py is the training code that uses sentencepieces for encoding and decoding.


It is recommended to use traine_scence-py for training, as the model trained in this way can be adapted to its tokenizer and model. And C++code is encoded and decoded using sentence pieces, so the trained model can be directly applied to myct. If you don't need to use C++for translation, then you can use whichever training you like.

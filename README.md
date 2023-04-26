[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a pytorch implementation for paper [Dual Contrastive Learning for Unsupervised Knowledge Selection](https://ksiresearch.org/seke/seke22paper/paper054.pdf)

1. Download

Download [bert_uncased_L-12_H-768_A-12](https://huggingface.co/google/bert_uncased_L-12_H-768_A-12/tree/main) and decompress it as "pretrained_bert\uncased_L-12_H-768_A-12".

Donwload [glove.840B.300d.txt](https://www.kaggle.com/datasets/takuok/glove840b300dtxt) and put it in "embeddings".

2. Train

python network.py --model=contrastive_seq2seq --alpha=0.05 --beta=0.05

3. Test

python network.py --test ----ckpt=ckpt-to-test
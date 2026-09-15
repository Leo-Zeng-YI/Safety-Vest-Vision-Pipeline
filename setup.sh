#!/bin/bash

# 1. 安装 Python 依赖包
pip install -r requirements.txt

# 2. SAM 3 缺失的词表文件
mkdir -p /usr/local/lib/python3.11/dist-packages/assets/
wget -O /usr/local/lib/python3.11/dist-packages/assets/bpe_simple_vocab_16e6.txt.gz https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz

echo "环境初始化与资产下载完成！"
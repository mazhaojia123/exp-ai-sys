#!/bin/bash

DATASET=./squad/dev-v1.1.json
VOCAB_FILE=./squad/vocab.txt

python3 pytorch_bert_inference.py --predict_file $DATASET \
                       --output_dir ./output \
                       --max_seq_length 128  \
                       --batch_size 16 \
                       --tokenizer_name $VOCAB_FILE

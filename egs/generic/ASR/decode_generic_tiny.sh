#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
./zipformer/streaming_decode.py \
    --epoch 24 \
    --avg 5 \
    --use-averaged-model 1 \
    --exp-dir zipformer/exp-en-tiny \
    --causal 1 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --chunk-size 128 \
    --left-context-frames 256 \
    --num-decode-streams 100 \
    --decoding-method greedy_search \
    --max-duration 500

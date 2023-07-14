#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

echo "Lhotse prep"
lhotse prepare generic -cj download/en-xx/generic_en-xx_mini.jsonl -od data/manifests -c en-xx_general_mini

echo "Lhotse splits"
python local/prepare_splits.py -p en-xx_general_mini

lang_dir=data/lang_bpe_500
mkdir -p $lang_dir

local/generate_transcript.py \
    -t data/manifests/en-xx_general_mini_supervisions_train.jsonl.gz \
    -t data/manifests/en-xx_general_mini_supervisions_dev.jsonl.gz \
    -o $lang_dir/transcript_words.txt \
    -w $lang_dir/words.txt

./local/train_bpe_model.py \
    --lang-dir $lang_dir \
    --vocab-size 500 \
    --transcript $lang_dir/transcript_words.txt

python3 local/compute_fbank_generic.py --test
python3 local/compute_fbank_generic.py --train --num-splits 40 

pieces=$(find data/manifests -name "cuts_train_[0-9]*.jsonl.gz")
lhotse combine $pieces data/manifests/cuts_train.jsonl.gz
gunzip -c data/manifests/cuts_train.jsonl.gz | shuf | gzip -c > data/manifests/cuts_train_shuf.jsonl.gz

# Large scale model
#  --num-encoder-layers 2,2,4,5,4,2 \
#  --feedforward-dim 512,768,1536,2048,1536,768 \
#  --encoder-dim 192,256,512,768,512,256 \
#  --encoder-unmasked-dim 192,192,256,320,256,192 \

# Normal (omit these options for normal scale)

# Small scale model
./zipformer/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-en-mini \
  --causal 1 \
  --max-duration 500 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,768,768,768,768 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --base-lr 0.04 \
  --chunk-size "16,32,64,128,-1" \
  --left-context-frames "64,128,256,512,-1"

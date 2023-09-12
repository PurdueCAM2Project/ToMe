#!/bin/bash
python eval.py --timm-model deit_small_patch16_224 --batch_size 1 --dataset imagenet1k --dataset-root /usr/local/data/ImageNet1K/ILSVRC/Data/CLS-LOC/ --num-workers 4

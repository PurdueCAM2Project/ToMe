#!/bin/bash
python eval.py --timm-model deit_base_patch16_224 --batch-size 32 --dataset imagenet1k --dataset-root /usr/local/data/ImageNet1K/ILSVRC/Data/CLS-LOC/ --num-workers 4

#!/bin/bash

conda activate <environment>
cd src

for K in {0..14}; do
    python train.py --experiment_root "../experiments/PURE_exper" --K $K --dataset "pure_unsupervised"
done

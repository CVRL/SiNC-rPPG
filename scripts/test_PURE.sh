#!/bin/bash

conda activate <environment>
cd src

python test.py --experiment_root "../experiments/PURE_exper"

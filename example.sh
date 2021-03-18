#!/bin/bash

dataset='2k'
t_0=0
T=8
t_train=8
K=2
epochs=10
H_0_npf='emb/H_0.2k.npy'

job_cmd="python main.py --dataset $dataset --t_0 $t_0 --T $T --t_train $t_train --K $K --epochs $epochs --H_0_npf $H_0_npf"

eval $job_cmd

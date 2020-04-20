#!/usr/bin/env bash
. ~/.bashrc
eval $(conda shell.bash hook)
conda activate topo2vec
tensorboard --logdir='/home/topo2vec_kavitzky/topo2vec/tb_logs/logs' --host=0.0.0.0 --port=$1

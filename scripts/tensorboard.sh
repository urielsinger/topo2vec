#!/usr/bin/env bash
. ~/.bashrc
conda activate topo2vec
tensorboard --logdir='/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/tb_logs/logs' --host=0.0.0.0 --port=$1

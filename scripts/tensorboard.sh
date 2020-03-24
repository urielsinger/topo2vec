#!/usr/bin/env bash
. ~/.bashrc
conda activate topo2vec
cd '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/tb_logs'
tensorboard --logdir=1classifier --host=0.0.0.0 --port=$1

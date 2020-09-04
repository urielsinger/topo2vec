#!/usr/bin/env bash
. ~/.bashrc
eval $(conda shell.bash hook)

conda activate topo2vec
cd '/home/topo2vec_kavitzky/topo2vec/notebooks'
jupyter notebook --no-browser --ip=0.0.0.0 --port=$1 --allow-root
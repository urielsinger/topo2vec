#!/usr/bin/env bash
. ~/.bashrc
conda activate topo2vec
cd '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/'

python 'topo2vec/experiments/'$1
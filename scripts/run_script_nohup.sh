#!/usr/bin/env bash
. ~/.bashrc
conda activate topo2vec
cd '/home/topo2vec_kavitzky/topo2vec/'

python 'topo2vec/experiments/'$1
#!/usr/bin/env bash
# delete all the tensorboard data
# should restart the tensorboard to work with properly
sudo rm -fr /home/topo2vec_kavitzky/topo2vec/tb_logs
mkdir /home/topo2vec_kavitzky/topo2vec/tb_logs
sudo chmod -R 777 /home/topo2vec_kavitzky/topo2vec/tb_logs
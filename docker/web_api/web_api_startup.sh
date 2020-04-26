#!/usr/bin/env bash
cd /home/topo2vec_kavitzky/topo2vec/server_api/
sudo apt-get update
sudo apt-get install gnome-terminal

#run flask
export FLASK_APP=server_api_instance.py
gnome-terminal --tab -- 'flask run --port=8765 --address=0.0.0.0'

cd /home/root/scripts
#run tensorboard
gnome-terminal -- 'tensorboard.sh 7777'

#run jupyter
gnome-terminal -- 'jupyter.sh 8888'
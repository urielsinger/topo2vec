#!/usr/bin/env bash
service ssh restart
bash /opt/conda/etc/profile.d/conda.sh init
cd /home/root/server_api/visualizations
bash conda activate topo2vec && conda info
bokeh serve bokeh_server --port=6543 --address=0.0.0.0 --allow-websocket-origin=127.0.0.1:6543

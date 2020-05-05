#!/usr/bin/env bash

echo 'starting'
service ssh restart
service ssh restart

#cd /home/root/api_server/visualizations
echo 'starting bokeh in port '$1
#bokeh serve bokeh_server --port=8765 --address=0.0.0.0 --allow-websocket-origin=127.0.0.1:8765 --allow-websocket-origin=159.122.160.130:8765
/opt/conda/envs/topo2vec/bin/bokeh serve /home/root/gui_server/bokeh_server \
        --port=$1 --address=0.0.0.0 --allow-websocket-origin=159.122.160.134:$1  --allow-websocket-origin=127.0.0.1:$1
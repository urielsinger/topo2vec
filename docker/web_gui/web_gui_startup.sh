eval "$(conda shell.bash hook)"
cd /home/root/server_api/visualizations/bokeh_server
#cd /home/topo2vec_kavitzky/topo2vec/server_api/visualizations/bokeh_server
conda activate topo2vec
#bokeh serve . --address=0.0.0.0 --port=5432 --allow-websocket-origin=0.0.0.0:5432
bash /opt/conda/etc/profile.d/conda.sh init
bash /opt/conda/etc/profile.d/conda.sh activate topo2vec
conda info
cd /home/root/server_api/visualizations
bokeh serve bokeh_server --port=5432 --address=0.0.0.0 --allow-websocket-origin=127.0.0.1:5432
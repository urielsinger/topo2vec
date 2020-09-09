cd /home/root/api_server/
service ssh restart
service ssh restart
#run flask
export FLASK_APP=server_api_instance.py
#run tensorboard
#tmux new-session 'cd /home/root/scripts && tensorboard.sh 7777'

/opt/conda/envs/topo2vec/bin/flask run --port=6666 --host=0.0.0.0

#run jupyter
#gnome-terminal -- 'jupyter.sh 8888'

cd /home/root/server_api/
service ssh restart

#run flask
export FLASK_APP=server_api_instance.py
#gnome-terminal --tab -- 'flask run --port=8765 --address=0.0.0.0'
flask run --port=6543 --host=0.0.0.0

#cd /home/root/scripts
#run tensorboard
#gnome-terminal -- 'tensorboard.sh 7777'

#run jupyter
#gnome-terminal -- 'jupyter.sh 8888'

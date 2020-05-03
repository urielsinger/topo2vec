# define the container itself: all installations etc.
FROM base_anaconda
EXPOSE 6543 6622
RUN apt-get update

ENTRYPOINT /bin/bash && conda init && conda activate topo2vec && bash /home/root/docker/web_api/web_api_startup.sh
ENTRYPOINT bash /home/root/docker/web_api/web_api_startup.sh
#run tensorboard in the right place

#run jupyter notebook in the right place

#run the flask server for getting programers api

#run the bokeh server for using the GUI

#run tests


# run with nvidia-docker (command installed via apt)







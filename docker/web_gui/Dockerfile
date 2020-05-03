# define the container itself: all installations etc.
FROM base_anaconda

USER root
WORKDIR /home/root

# expose for the bokeh server
EXPOSE 8765 7722

#SHELL ["conda", "run", "-n", "topo2vec", "/bin/bash", "-c"]

#run the bokeh server for using the GUI
#ENTRYPOINT ["bash", "/home/root/docker/web_gui/web_gui_startup.sh"]
ENTRYPOINT ["conda", "run", "-n", "topo2vec", "bash", "/home/root/docker/web_gui/web_gui_startup.sh"]
ENTRYPOINT bash /home/root/docker/web_gui/web_gui_startup.sh
#CMD /bin/bash && conda init && conda activate topo2vec && bokeh
#run tensorboard in the right place

#run jupyter notebook in the right place

#run tests

# run with nvidia-docker (command installed via apt)







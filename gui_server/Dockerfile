# define the container itself: all installations etc.
FROM base_anaconda

USER root
WORKDIR /home/root

# expose for the bokeh server
EXPOSE ${BOKEH_PORT} 7722

#SHELL ["conda", "run", "-n", "topo2vec", "/bin/bash", "-c"]

#run the bokeh server for using the GUI
#ENTRYPOINT ["conda", "run", "-n", "topo2vec", "bash", "/home/root/docker/web_gui/web_gui_startup.sh"]
RUN echo 'running in port' ${BOKEH_PORT}
ENTRYPOINT bash /home/root/gui_server/web_gui_startup.sh ${BOKEH_PORT}
#CMD /bin/bash && conda init && conda activate topo2vec && bokeh
#run tensorboard in the right place

#run jupyter notebook in the right place

#run tests

# run with nvidia-docker (command installed via apt)







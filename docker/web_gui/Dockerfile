# define the container itself: all installations etc.
FROM base_anaconda

USER root
WORKDIR /home/root

# expose for the bokeh server
EXPOSE 5432

#configure ssh
RUN apt-get update
RUN apt-get install -y openssh-server gedit

RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN echo 'root:Docker!' | chpasswd

RUN service ssh restart
RUN service ssh restart


#run the bokeh server for using the GUI
#CMD ["bash", "/home/root/docker/web_gui/web_gui_startup.sh"]
CMD ["bash"]

#run tensorboard in the right place

#run jupyter notebook in the right place
#CMD python /app/app.py

#run the flask server for getting programers api


#run tests


# run with nvidia-docker (command installed via apt)







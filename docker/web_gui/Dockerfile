# define the container itself: all installations etc.
FROM base_anaconda

USER root
WORKDIR /home/root
EXPOSE 5432

#run the bokeh server for using the GUI
RUN /bin/sh -c "cd /home/root/server_api/visualizations; bokeh serve --address=0.0.0.0 --port=5432"


#run tensorboard in the right place

#run jupyter notebook in the right place
#CMD python /app/app.py

#run the flask server for getting programers api


#run tests


# run with nvidia-docker (command installed via apt)







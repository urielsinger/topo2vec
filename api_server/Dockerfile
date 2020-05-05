# define the container itself: all installations etc.
FROM base_anaconda
ENV FLASK_PORT 6543
EXPOSE ${FLASK_PORT} 6622
RUN echo 'building with port' ${FLASK_PORT}
ENTRYPOINT bash /home/root/api_server/web_api_startup.sh ${FLASK_PORT}
#run tensorboard in the right place

#run jupyter notebook in the right place

#run the flask server for getting programers api

#run the bokeh server for using the GUI

#run tests

# run with nvidia-docker (command installed via apt)







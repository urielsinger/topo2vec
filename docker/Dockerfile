# define the container itself: all installations etc.
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
COPY .. /app

# Set up environment and renderer user
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install useful commands
RUN apt-get update
RUN apt-get install software-properties-common -y

# Install topo2vec environment
WORKDIR /home/root
RUN curl https://gist.githubusercontent.com/jonzarecki/bef39acdd631f6cbba6ce9b04b539138/raw/ef4e5542409259d871189978495bee271b75ad53/geo_envffile.yml > environment.yml
RUN conda env create -f environment.yml
RUN conda init bash

# Init coord2vec environment
ENV PATH /opt/conda/envs/coord2vec/bin:$PATH
RUN /bin/bash -c "source activate topo2vec"

# Start running
USER root
WORKDIR /home/root
# Init coord2vec environment
#RUN echo "conda activate coord2vec" >> ~/.basrhc
#ENV PATH /opt/conda/envs/coord2vec/bin:$PATH
#RUN /bin/bash -c "source activate coord2vec"
#RUN conda init bash
#RUN conda activate coord2vec
ENTRYPOINT ["/bin/bash"]
CMD []

#run tensorboard in the right place

#run jupyter notebook in the right place
CMD python /app/app.py

#run the flask server for getting programers api

#run the bokeh server for using the GUI

#run tests


# run with nvidia-docker (command installed via apt)







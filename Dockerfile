# define the container itself: all installations etc.
FROM pytorch/pytorch:latest

# Set up environment and renderer user
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install useful commands
RUN apt-get update
RUN apt-get install software-properties-common -y
RUN apt install -y curl

# Install topo2vec environment
WORKDIR /home/root
RUN curl https://gist.githubusercontent.com/johnnyrunner/9c0ab0ffd24301ea2fcc5a377b9b9448/raw/35b9711ca7fbef899d7ba4e91312dcc2a2e2b37a/topo_envffile.yml > environment.yml
RUN conda update -y -n base -c defaults conda
RUN conda env create -f environment.yml
RUN conda init bash

#fix stuff about the usage of opencv
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python

# Init topo2vec environment
ENV PATH /opt/conda/envs/topo2vec/bin:$PATH
RUN /bin/bash -c "source activate topo2vec"

# Start running
USER root
WORKDIR /home/root

RUN /bin/bash -c "mkdir -p topo2vec"

COPY . .

EXPOSE 80

#configure ssh
RUN apt-get update
RUN apt-get install -y openssh-server gedit gnome-terminal

RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN echo 'root:Docker!' | chpasswd




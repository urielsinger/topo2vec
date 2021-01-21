Topo2vec: Topography Embedding Using the Fractal Effect
====
This repository provides an implementation of *Topo2vec*, all baselines, and code for datasets building and experiments, as described in the paper:<br>
> Topo2vec: Topography Embedding Using the Fractal Effect<br>

The *Topo2vec* algorithm is the first embedding technique for topographic data, based on the fractal effect.
![Topo2vec](architecture.jpg)


## Build the environment
 - compose the full system (including gui server and api server) using docker-compose and the docker-compose.yml file.
 1. define the wanted ports for the project in the .env file
  ```angular2
docker-compose build --no-cache
```
 3. in the root folder run:
  ```angular2
docker compose up
```
 OR ANOTHER WAY:
 - use the docker image provided in the Dockerfile
 ```angular2
docker run -it -v .:/home/root/. -p ${API_SSH_PORT}:22 --env-file .env --gpus all --name web_gpu web_api2 /bin/bash
```
and inside the container:
```
activate topo2vec, uninstall torch, and run: "pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html"
```
 OR ANOTHER WAY:
 - follow the environment.yml file and try to build it locally (this may not work according to your machine state and is not recommended)


## Download the data
a. download the needed data:
1. run the "scripts/get_elevation_data.sh min_lon, min_lat, max_lon, max_lat"
2. edit and run the "scripts/get_classes_data.sh min_lon, min_lat, max_lon, max_lat"
3. put all the classes' test data in the data/overpass_classes_data/tests in the needed json format (the same one as in the train)

It is much recommended to download the the data inside the docker container.

## Usage
1. For exploring the experiments in the paper, first download the data (as in the "download the data" section), 
then train the appropriate models you want - using the training files inside the "experiments" folder (it is recommended to use tensorboard),
and then run the python files inside "evaluation experiments"
 
2. Using the GUI server for exploration - go to the gui's container address:BOKEH_PORT (as in the .env file)
  

## Cite
Please cite our paper if you use this code in your own work:

```
@inproceedings{topo2vec,
  title={Topo2vec: Topography Embedding Using the Fractal Effect}
}
```
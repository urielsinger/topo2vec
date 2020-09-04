# initial setup of the project:
a. download the needed data:
1. run the "scripts/get_elevation_data.sh min_lon, min_lat, max_lon, max_lat"
2. edit and run the "scripts/get_classes_data.sh min_lon, min_lat, max_lon, max_lat"
3. put all the classes test data in the data/overpass_classes_data/tests in the needed json format

b. run the project:
1. define the wanted ports for the project in the .env file
2. in the root folder run "docker-compose build --no-cache"
3. in the root folder run "docker compose up"
4. if want to run a single docker: "docker run -it -v .:/home/root/. -p ${API_SSH_PORT}:22 --env-file .env --gpus all --name web_gpu web_api2 /bin/bash"
5. inside docker, activate topo2vec, uninstall torch, and run: "pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html"

c. run additional debugging tools:
1. run tensorboard using "scripts/tensorboard.sh port_num"
    - add the needed ports to the docker-compose file.
2. run jupyter notebook using "scripts/jupyter.sh port_num"
    - add the needed ports to the docker-compose file.

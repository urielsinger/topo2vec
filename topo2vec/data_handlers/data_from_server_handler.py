import logging
import requests
import os
from io import BytesIO
from PIL import Image, ImageTk
import numpy as np

from topo2vec.data_handlers.data_handler import DataHandler


class DataFromServerHandler(DataHandler):
    def __init__(self, server_ip='localhost', port='8080'):
        '''

        Args:
            server_ip: The remote server's ip
            port: the open port, inside the docker.
        '''
        self.ip = server_ip
        self.port = port
        self.base_url = f'http://{server_ip}:{port}/{{lon}}/{{lat}}/{{r}}/tile.{{dtype}}'

    def get_patch_response_from_server(self, lon, lat, radius, dtype='tiff', retries=5):
        '''
        Download some content from the server
        Args:
            lon:
            lat:
            radius:
            dtype:
            retries: number of times to try and fetch the data

        Returns: the Response containing the data/img asked

        '''
        url = self.base_url.format(lon=lon, lat=lat, r=radius, dtype=dtype)

        for retry in range(retries):
            response = requests.get(url)
            if response.status_code == 200:
                break
            else:
                logging.info(f'failed to fetch {url}, retry number {retry}'
                             f'response.status_code = {response.status_code}')
        return response

    def write_patch_to_file(self, file_name, directory='', file_type='data', response=None):
        '''

        Args:
            file_name:
            directory:
            file_type: what kind of file is it
            response: The reponse the file came from

        Returns:

        '''
        if response != None:
            if file_type == 'data':
                file_name += '.tiff'
            elif file_type == 'img':
                file_name += '.png'
            path = os.path.join(directory, '../data/' + file_name)
            f = open(path, 'wb+')
            f.write(response.content)
            f.close()
            return True  # worked

    def get_data_as_np_array_from_response(self, response):
        '''
        Must be used using a tiff file came from the server
        Args:
            response: The reponse the data came from

        Returns: an np array containing the data

        '''
        img = Image.open(BytesIO(response.content))
        img_as_array = np.asarray(img)
        return img_as_array

    def get_data_as_np_array(self, lon, lat, r, dtype='tiff', retries=5):
        '''
        get the data from the server
        Args:
            lon:
            lat:
            r:
            dtype:
            retries:

        Returns: the data as an np array

        '''
        response = self.get_patch_response_from_server(lon, lat, r, dtype, retries)
        return self.get_data_as_np_array_from_response(response)

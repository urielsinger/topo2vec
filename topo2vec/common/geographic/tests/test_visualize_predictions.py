import unittest

from coord2vec.common.geographic.visualizations import visualize_predictions
from flask import Flask
import numpy as np

app = Flask(__name__)


class TestMapVisualizations(unittest.TestCase):
    def test_flask_running(self):
        self.skipTest("need to get to link and check manually")
        l = None
        map = visualize_predictions(np.array([(l[0]+i*0.005, l[1]+j*0.005) for i in range(50) for j in range(50)]),
                              np.random.random((50*50,)) - 0.5)

        @app.route('/')
        def index():
            # start_coords = (46.9540700, 142.7360300)
            # folium_map = folium.Map(location=start_coords, zoom_start=14)
            return map._repr_html_()

        app.run(debug=True)
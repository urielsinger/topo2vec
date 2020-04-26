import os
import random
import sys

import numpy

from bokeh.io import curdoc
from pathlib import Path

my_path = os.path.abspath(__file__)
parent_path = Path(my_path).parent.parent.parent.parent
sys.path.append(str(parent_path))
print(parent_path)
from server_api.visualizations.bokeh_server.basic_bokeh import BasicBokeh


def main():
    random.seed(42)
    numpy.random.seed(42)
    task = None
    # create bokeh tabs
    BokehData = BasicBokeh()

    curdoc().add_root(BokehData.main_panel)
    curdoc().title = "bokeh for choosing points"

main()
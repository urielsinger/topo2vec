from flask import Flask

import folium

app = Flask(__name__)

from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure

@app.route('/')
def index():
    start_coords = (46.9540700, 142.7360300)
    folium_map = folium.Map(location=start_coords, zoom_start=14)
    folium_map.add_child(folium.ClickForMarker(popup="new class chosen"))
    sliders = column(amp, freq, phase, offset)
    return folium_map._repr_html_()


if __name__ == '__main__':
    app.run(debug=True)
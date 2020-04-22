''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import os

import folium
import numpy as np
import time

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, Div, Select, Button, Text
from bokeh.plotting import figure
from shapely.geometry import Polygon, Point

import sys
from pathlib import Path

my_path = os.path.abspath(__file__)
parent_path = Path(my_path).parent.parent.parent.parent
sys.path.append(str(parent_path))

from topo2vec.modules.visual_topography_profiler import TopoMap

BASIC_POLYGON = Polygon([Point(5, 45), Point(5, 45.1), Point(5.1, 45.1),
                         Point(5.1, 45.1), Point(5, 45)])

points_inside = [Point(5.0658811, 45.0851164),
                 Point(5.058811, 45.01164)]


class BasicBokeh:
    def __init__(self):
        # Set up data
        self.N = 200
        x = np.linspace(0, 4 * np.pi, self.N)
        y = np.sin(x)
        self.source = ColumnDataSource(data=dict(x=x, y=y))

        # Set up map
        # self.plot = figure(plot_height=400, plot_width=400, title="my sine wave",
        #                    tools="crosshair,pan,reset,save,wheel_zoom",
        #                    x_range=[0, 4 * np.pi], y_range=[-2.5, 2.5])

        self.topo_map = TopoMap(BASIC_POLYGON)
        self.folium_fig = self.bokeh_new_class_folium()

        # self.plot.line('x', 'y', source=self.source, line_width=3, line_alpha=0.6)

        # Set up widgets
        text = TextInput(title="title", value='my sine wave')
        self.offset = Slider(title="self.offset", value=0.0, start=-5.0, end=5.0, step=0.1)
        self.amplitude = Slider(title="self.amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
        self.phase = Slider(title="self.phase", value=0.0, start=0.0, end=2 * np.pi)
        self.freq = Slider(title="self.frequency", value=1.0, start=0.1, end=5.1, step=0.1)

        lon_text = TextInput(value="5.05", title='lon:', width=100)
        lat_text = TextInput(value="45.05", title='lat:', width=100)
        self.lonlat_text_inputs = [lon_text, lat_text]
        get_points_button = Button(label='Get desired points!')
        get_points_button.on_click(self.get_points_and_update)

        text.on_change('value', self.update_title)

        for w in [self.offset, self.amplitude, self.phase, self.freq]:
            w.on_change('value', self.update_data)

        # Set up layouts and add to document
        inputs = column(row(lon_text, lat_text), get_points_button,
                        text, self.offset, self.amplitude, self.phase, self.freq)
        self.main_panel = row(inputs, self.folium_fig, width=800)

    def bokeh_new_class_folium(self, file_name: str = 'folium',
                               lonlat_text_inputs: list = None,
                               height: int = 600, width: int = 850):
        # topo_map.add_points_with_text(points_chosen, text='original images')
        # folium.LayerControl().add_to(self.topo_map.map)
        # folium.plugins.MeasureControl().add_to(topo_map.map)
        # folium.plugins.MousePosition(lng_first=True).add_to(topo_map.map)
        # self.topo_map.map.add_child(folium.ClickForMarker(popup="new class chosen"))

        # save the folium html
        static_folder = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_folder, exist_ok=True)
        file_name = f'{hash(file_name)}{str(self.topo_map.version)}.html'
        # self.topo_map.add_all_class_points(BASIC_POLYGON, 500, 'peaks')
        filePath = os.path.join(static_folder, file_name)

        if os.path.exists(filePath):
            print('removing')
            os.remove(filePath)
        self.topo_map.map.save(filePath)

        click_str = f"""
                    f.contentWindow.document.body.onclick = 
                    function() {{
                        ff = document.getElementById('{hash(file_name)}');
                        map = eval('ff.contentWindow.'+ff.contentWindow.document.getElementsByClassName('folium-map')[0].id);
                        window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{
        lonlat_text_inputs[0].id}].value = map.loc[1].toFixed(5).toString();
                        window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{
        lonlat_text_inputs[1].id}].value = map.loc[0].toFixed(5).toString();
                    }};
                    """ if lonlat_text_inputs is not None else ""
        fig = Div(text=f"""
        <iframe onload="console.log('changing map props');
                    f = document.getElementById('{hash(file_name)}');
                    map = eval('f.contentWindow.'+f.contentWindow.document.getElementsByClassName('folium-map')[0].id);
                        if(self.center && self.zoom){{map.setView(self.center, self.zoom);}};

                    {click_str}
                    "

                id="{hash(file_name)}"
                src="bokeh_server/static/{file_name}"
                width={width} height={height}></iframe>
        """, height=height, width=width)
        return fig

    def get_points_and_update(self):
        lat, lon = self.get_click_lonlat()
        point = Point(lon, lat)

        points_chosen = [point]
        self.topo_map.add_similar_points(points_chosen, polygon=BASIC_POLYGON,
                                         meters_step=500, n=10, color='red')
        self.topo_map.add_points_with_text(points_chosen, color='green', text='chosen')
        self.topo_map.add_all_class_points(BASIC_POLYGON, 500, 'peaks')
        self.topo_map.version += 1
        fig = self.bokeh_new_class_folium(lonlat_text_inputs = [])
        self.main_panel.children[-1] = fig

        # self.lonlat_text_inputs[0].value = str(round(closest_geo.centroid.x, 5))
        # self.lonlat_text_inputs[1].value = str(round(closest_geo.centroid.y, 5))
        # self.importance_and_val_column.children[-1] = importance_figure

    def get_click_lonlat(self):
        lon = self.lonlat_text_inputs[0].value
        lat = self.lonlat_text_inputs[1].value
        lon = float(lon) if lon != "" else 0
        lat = float(lat) if lat != "" else 0
        return lat, lon

    def update_title(self, attrname, old, new):
        # Set up callbacks
        self.plot.title.text = self.text.value

    def update_data(self, attrname, old, new):
        # Get the current slider values
        a = self.amplitude.value
        b = self.offset.value
        w = self.phase.value
        k = self.freq.value

        # Generate the new curve
        x = np.linspace(0, 4 * np.pi, self.N)
        y = a * np.sin(k * x + w) + b

        self.source.data = dict(x=x, y=y)

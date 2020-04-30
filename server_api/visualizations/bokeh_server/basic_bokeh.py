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
from bokeh.models import ColumnDataSource, Slider, TextInput, Div, Select, Button, Text, Dropdown
from bokeh.plotting import figure
from shapely.geometry import Polygon, Point

import sys
from pathlib import Path

from topo2vec import visualizer
from topo2vec.common.visualizations import convert_multi_radius_ndarray_to_printable, \
    convert_multi_radius_list_to_printable
from topo2vec.modules import visual_topography_profiler

my_path = os.path.abspath(__file__)
parent_path = Path(my_path).parent.parent.parent.parent
sys.path.append(str(parent_path))

from topo2vec.modules.visual_topography_profiler import TopoMap

points_inside = [Point(5.0658811, 45.0851164),
                 Point(5.058811, 45.01164)]

WORKING_POLYGON = visual_topography_profiler._get_working_polygon()


class BasicBokeh:
    start_location = visual_topography_profiler._get_working_polygon_center()

    def __init__(self):
        self.zoom = 12
        self.center = self.start_location

        # Set up map
        lon_text = TextInput(value='', title='lon:')
        lat_text = TextInput(value='', title='lat:')
        self.lonlat_text_inputs = [lon_text, lat_text]

        self.topo_map = TopoMap(WORKING_POLYGON)
        self.folium_fig = self.bokeh_new_class_folium(lonlat_text_inputs=self.lonlat_text_inputs)

        # self.plot.line('x', 'y', source=self.source, line_width=3, line_alpha=0.6)

        # Set up widgets
        self.meters_step = Slider(title="meters_step", value=500, start=50, end=1000, step=10)
        self.number_of_points_to_show = Slider(title="number of points to show", value=5, start=1, end=100)
        # self.threshold = Slider(title="threshold for class", value=0.5, start=0, end=1000, step=10)

        get_points_button = Button(label='Get desired points!')
        get_points_button.on_click(self.get_points_and_update)

        clean_all_buttun = Button(label='clean all')
        clean_all_buttun.on_click(self.clean_all)

        clean_text_buttun = Button(label='clean text')
        clean_text_buttun.on_click(self.clean_text)

        select_class_options = self.topo_map.get_all_available_classes()
        self.select = Select(title="Option:", value=select_class_options[0], options=select_class_options)

        get_class_button = Button(label='get_class')
        get_class_button.on_click(self.get_points_and_update_for_class)

        # Set up layouts and add to document
        inputs = row(
            column(Div(text='get similar points'), lon_text, lat_text, get_points_button, clean_all_buttun,
                   clean_text_buttun),
            column(Div(text='get top points of class'), self.select, get_class_button),
            column(Div(text='search parameters'), self.meters_step, self.number_of_points_to_show))
        self.main_panel = row(inputs, self.folium_fig, width=800)

    def bokeh_new_class_folium(self, file_name: str = 'folium',
                               lonlat_text_inputs: list = None,
                               height: int = 600, width: int = 850):
        # topo_map.add_points_with_text(points_chosen, text='original images')
        # folium.LayerControl().add_to(self.topo_map.map)
        # folium.plugins.MeasureControl().add_to(topo_map.map)
        # folium.plugins.MousePosition(lng_first=True).add_to(topo_map.map)
        self.topo_map.map.add_child(folium.ClickForMarker(popup="new class chosen"))
        self.topo_map.map.add_child(folium.LatLngPopup())

        # save the folium html
        static_folder = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_folder, exist_ok=True)
        file_name_hash = hash(f'{file_name}{time.time()}')
        file_name = f'{file_name_hash}.html'
        filePath = os.path.join(static_folder, file_name)

        if os.path.exists(filePath):
            print('removing')
            os.remove(filePath)

        self.topo_map.map.save(filePath)

        click_str = f"""
                    f.contentWindow.document.body.onclick = 
                    function() {{
                        ff = document.getElementById('{file_name_hash}');
                        popup = ff.contentWindow.document.getElementsByClassName('leaflet-popup-content')[0];
                        popup_text = popup.innerHTML
                        if(popup_text.length==38||popup_text.length==39){{
                        popup_words = popup_text.split(' ')
                        longtitude = popup_words[2]
                        latitude = popup_words[1].split('<br>')[0]
                        console.log(longtitude);
                        console.log(latitude);
                        lon_old_values = window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{lonlat_text_inputs[0].id}].value
                        window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{lonlat_text_inputs[0].id}].value = longtitude + ', ' +  lon_old_values;

                        lat_old_values = window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{lonlat_text_inputs[1].id}].value;
                        window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{lonlat_text_inputs[1].id}].value = latitude + ', ' +  lat_old_values;
                        }}
                    }};
                    """ if lonlat_text_inputs is not None else ""
        fig = Div(text=f"""
        <iframe onload="console.log('changing map props');
                    f = document.getElementById('{file_name_hash}');
                    map = eval('f.contentWindow.'+f.contentWindow.document.getElementsByClassName('folium-map')[0].id);
                        if({self.center} && {self.zoom}){{console.log('lala'); map.setView({self.center}, {self.zoom});}};

                    {click_str}
                    "

                id="{file_name_hash}"
                src="bokeh_server/static/{file_name}"
                width={width} height={height}></iframe>
        """, height=height, width=width)
        return fig

    def get_points_and_update(self):
        self.topo_map = TopoMap(WORKING_POLYGON)
        points_chosen = self.get_click_lonlat_points_list()
        self.topo_map.add_similar_points(points_chosen, polygon=WORKING_POLYGON,
                                         meters_step=int(self.meters_step.value),
                                         n=int(self.number_of_points_to_show.value), color='red')
        images, _ = visualizer.get_points_as_list_of_np_arrays(points_chosen, [8, 16, 24]) #TODO: change the const!!
        points_images = convert_multi_radius_list_to_printable(images, dir=False)
        self.topo_map.add_points_with_images(points_chosen, points_images, color='green')
        # self.topo_map.add_points_with_text(points_chosen, color='green', text='chosen')
        fig = self.bokeh_new_class_folium(lonlat_text_inputs=self.lonlat_text_inputs)
        self.main_panel.children[-1] = fig

        # self.lonlat_text_inputs[0].value = str(round(closest_geo.centroid.x, 5))
        # self.lonlat_text_inputs[1].value = str(round(closest_geo.centroid.y, 5))
        # self.importance_and_val_column.children[-1] = importance_figure

    def get_points_and_update_for_class(self):
        self.topo_map = TopoMap(WORKING_POLYGON)
        class_chosen = self.select.value
        self.topo_map.add_random_class_points(polygon=WORKING_POLYGON,
                                              meters_step=int(self.meters_step.value), class_name=class_chosen,
                                              color='red', max_num=int(self.number_of_points_to_show.value))

        fig = self.bokeh_new_class_folium(lonlat_text_inputs=self.lonlat_text_inputs)
        self.main_panel.children[-1] = fig

        # self.lonlat_text_inputs[0].value = str(round(closest_geo.centroid.x, 5))
        # self.lonlat_text_inputs[1].value = str(round(closest_geo.centroid.y, 5))
        # self.importance_and_val_column.children[-1] = importance_figure

    def clean_all(self):
        self.topo_map = TopoMap(WORKING_POLYGON)
        fig = self.bokeh_new_class_folium(lonlat_text_inputs=self.lonlat_text_inputs)
        self.main_panel.children[-1] = fig
        self.lonlat_text_inputs[0].value = ''
        self.lonlat_text_inputs[1].value = ''

    def clean_text(self):
        self.lonlat_text_inputs[0].value = ''
        self.lonlat_text_inputs[1].value = ''

    def get_click_lonlat_points_list(self):
        lon_list = self.lonlat_text_inputs[0].value.split(', ')
        lat_list = self.lonlat_text_inputs[1].value.split(', ')
        points_list = []
        for lon, lat in zip(lon_list, lat_list):
            if lon != "" and lat != "":
                lon = float(lon) if lon != "" else 0
                lat = float(lat) if lat != "" else 0
                point = Point(lon, lat)
                points_list.append(point)
        return points_list

    # def get_click_lonlat(self):
    #     lon = self.lonlat_text_inputs[0].value
    #     lat = self.lonlat_text_inputs[1].value
    #     lon = float(lon) if lon != "" else 0
    #     lat = float(lat) if lat != "" else 0
    #     return lat, lon

    def update_title(self, attrname, old, new):
        # Set up callbacks
        self.plot.title.text = self.text.value

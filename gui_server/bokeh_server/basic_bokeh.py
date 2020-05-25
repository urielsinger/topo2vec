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
import time

from bokeh.layouts import column, row
from bokeh.models import Slider, TextInput, Div, Select, Button
from shapely.geometry import Point, Polygon

import sys
from pathlib import Path

from api_client import client_lib
from common.geographic.geo_utils import build_polygon
from visualization_server import visualizer

from common.pytorch.visualizations import convert_multi_radius_list_to_printable
from gui_server import visual_topography_profiler

my_path = os.path.abspath(__file__)
parent_path = Path(my_path).parent.parent.parent
sys.path.append(str(parent_path))

from gui_server.visual_topography_profiler import TopoMap

points_inside = [Point(5.0658811, 45.0851164),
                 Point(5.058811, 45.01164)]

small_polygon = build_polygon(35.3, 33.11, 35.35, 33.15)


def set_working_polygon(polygon: Polygon):
    client_lib.set_working_polygon(polygon)
    global WORKING_POLYGON
    WORKING_POLYGON = visual_topography_profiler._get_working_polygon()

goral_hights = build_polygon(34.7, 31.3, 34.9, 31.43)
north_is = build_polygon(35.1782, 32.8877, 35.5092,  33.0524)
north_is_small = build_polygon(35.3782, 32.9877, 35.4092,  33.0000)

set_working_polygon(north_is_small)


class BasicBokeh:
    start_location = visual_topography_profiler.get_working_polygon_center()

    def __init__(self):
        self.zoom = 12
        self.center = self.start_location

        # Set up map
        lon_text = TextInput(value='', title='lon:')
        lat_text = TextInput(value='', title='lat:')
        self.lonlat_text_inputs = [lon_text, lat_text]

        self.topo_map = TopoMap(WORKING_POLYGON)
        self.folium_fig = self.bokeh_new_class_folium(lonlat_text_inputs=self.lonlat_text_inputs)

        # Set up widgets
        self.meters_step = Slider(title="meters_step", value=200, start=10, end=2000, step=10)
        self.number_of_points_to_show = Slider(title="number of points to show", value=5, start=1, end=100)
        self.threshold = Slider(title="threshold for class", value=0, start=0, end=1, step=0.01)
        self.test_minimal_resolution = Slider(title="minimal resolution", value=8, start=2, end=50)

        get_points_button = Button(label='Get desired points!')
        get_points_button.on_click(self.get_points_and_update)

        set_working_polygon_button = Button(label='Set working polygon!')
        set_working_polygon_button.on_click(self.set_working_polygon)

        clean_all_buttun = Button(label='clean all')
        clean_all_buttun.on_click(self.clean_all)

        clean_text_buttun = Button(label='clean text')
        clean_text_buttun.on_click(self.clean_text)

        select_class_options = self.topo_map.get_all_available_classes()
        self.select_class = Select(title="Option:", value=select_class_options[0], options=select_class_options)

        get_class_button = Button(label='get_class')
        get_class_button.on_click(self.get_points_and_update_for_class)

        select_final_model_options = client_lib.get_available_final_model_file_names()
        self.select_final_model = Select(title="Option:", value=select_final_model_options[0], options=select_final_model_options)

        final_model_button = Button(label='select final model')
        final_model_button.on_click(self.update_final_model)

        get_segmentation_map_button = Button(label='get segmentation map')
        get_segmentation_map_button.on_click(self.add_segmentation_map)

        self.row_mid_column = column(Div(text='get top points of class'), self.select_class, get_class_button,
                                     get_segmentation_map_button, self.select_final_model, final_model_button, Div(text=''))
        # Set up layouts and add to document
        inputs = row(
            column(Div(text='get similar points'), lon_text, lat_text, get_points_button, set_working_polygon_button, clean_all_buttun, clean_text_buttun),
            self.row_mid_column,
            column(Div(text='search parameters'), self.meters_step, self.number_of_points_to_show, self.threshold, self.test_minimal_resolution))

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
                        lon_old_values = window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{
        lonlat_text_inputs[0].id}].value
                        window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{
        lonlat_text_inputs[0].id}].value = longtitude + ', ' +  lon_old_values;

                        lat_old_values = window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{
        lonlat_text_inputs[1].id}].value;
                        window.Bokeh.index[Object.keys(window.Bokeh.index)[0]].model.document._all_models[{
        lonlat_text_inputs[1].id}].value = latitude + ', ' +  lat_old_values;
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
                                         n=int(self.number_of_points_to_show.value), color='red',
                                         test_radius=int(self.test_minimal_resolution.value))
        images, _ = visualizer.get_points_as_list_of_np_arrays(points_chosen, [8, 16, 24])  # TODO: change the const!!
        points_images = convert_multi_radius_list_to_printable(images, dir=False)
        self.topo_map.add_points_with_images(points_chosen, points_images, color='green')
        # self.topo_map.add_points_with_text(points_chosen, color='green', text='chosen')
        fig = self.bokeh_new_class_folium(lonlat_text_inputs=self.lonlat_text_inputs)
        self.main_panel.children[-1] = fig

        # self.lonlat_text_inputs[0].value = str(round(closest_geo.centroid.x, 5))
        # self.lonlat_text_inputs[1].value = str(round(closest_geo.centroid.y, 5))
        # self.importance_and_val_column.children[-1] = importance_figure

    def set_working_polygon(self):
        points_chosen = self.get_click_lonlat_points_list()
        new_working_polygon = Polygon(points_chosen + [points_chosen[0]])
        set_working_polygon(new_working_polygon)
        self.topo_map = TopoMap(WORKING_POLYGON)
        self.start_location = visual_topography_profiler.get_working_polygon_center()
        self.center = self.start_location
        fig = self.bokeh_new_class_folium(lonlat_text_inputs=self.lonlat_text_inputs)
        self.main_panel.children[-1] = fig

    def set_test_radius(self):
        resolution_decided=int(self.test_minimal_resolution.value)
        #set_test_radius(three_resolutions)
        self.topo_map = TopoMap(WORKING_POLYGON)
        self.start_location = visual_topography_profiler.get_working_polygon_center()
        self.center = self.start_location
        fig = self.bokeh_new_class_folium(lonlat_text_inputs=self.lonlat_text_inputs)
        self.main_panel.children[-1] = fig

    def add_segmentation_map(self):
        thresholds_list = [0, 0, 0, 0]
        class_names_list = client_lib.get_available_class_names()
        self.topo_map.add_segmentation_map(polygon=WORKING_POLYGON, meters_step=int(self.meters_step.value),
                                           class_names=class_names_list,
                                           thresholds_list=thresholds_list,
                                           test_radius=int(self.test_minimal_resolution.value))

        fig = self.bokeh_new_class_folium(lonlat_text_inputs=self.lonlat_text_inputs)
        self.main_panel.children[-1] = fig
        self.row_mid_column.children[-1] = Div(text=self.topo_map.colors_map) #TODO: very long


    def get_points_and_update_for_class(self):
        self.topo_map = TopoMap(WORKING_POLYGON)
        class_chosen = self.select_class.value
        self.topo_map.add_random_class_points(polygon=WORKING_POLYGON,
                                              meters_step=int(self.meters_step.value), class_name=class_chosen,
                                              color='red', max_num=int(self.number_of_points_to_show.value),
                                              threshold=float(self.threshold.value),
                                              test_radius=int(self.test_minimal_resolution.value))

        fig = self.bokeh_new_class_folium(lonlat_text_inputs=self.lonlat_text_inputs)
        self.main_panel.children[-1] = fig

        # self.lonlat_text_inputs[0].value = str(round(closest_geo.centroid.x, 5))
        # self.lonlat_text_inputs[1].value = str(round(closest_geo.centroid.y, 5))
        # self.importance_and_val_column.children[-1] = importance_figure

    def update_final_model(self):
        model_chosen = self.select_final_model.value
        client_lib.set_final_model(model_chosen)

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

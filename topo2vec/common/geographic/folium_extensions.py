import json
import warnings
import functools
import operator

from branca.colormap import LinearColormap, StepColormap
from branca.element import (Element, Figure, JavascriptLink, MacroElement, CssLink)
from branca.utilities import color_brewer
from folium import GeoJson, TopoJson

from folium.folium import Map
from folium.map import (FeatureGroup, Icon, Layer, Marker, Tooltip)
from folium.plugins import TimeSliderChoropleth, Search, MarkerCluster
from folium.utilities import (
    validate_locations,
    _parse_size,
    get_bounds,
    image_to_url,
    none_max,
    none_min,
    get_obj_in_upper_tree,
    parse_options,
)
from folium.vector_layers import PolyLine, path_options

from jinja2 import Template

import numpy as np

import requests


class NoClickGeoJson(GeoJson):
    _template = Template(u"""
        {% macro script(this, kwargs) %}
        {%- if this.style %}
        function {{ this.get_name() }}_styler(feature) {
            switch({{ this.feature_identifier }}) {
                {%- for style, ids_list in this.style_map.items() if not style == 'default' %}
                {% for id_val in ids_list %}case {{ id_val|tojson }}: {% endfor %}
                    return {{ style }};
                {%- endfor %}
                default:
                    return {{ this.style_map['default'] }};
            }
        }
        {%- endif %}
        {%- if this.highlight %}
        function {{ this.get_name() }}_highlighter(feature) {
            switch({{ this.feature_identifier }}) {
                {%- for style, ids_list in this.highlight_map.items() if not style == 'default' %}
                {% for id_val in ids_list %}case {{ id_val|tojson }}: {% endfor %}
                    return {{ style }};
                {%- endfor %}
                default:
                    return {{ this.highlight_map['default'] }};
            }
        }
        {%- endif %}
        function {{this.get_name()}}_onEachFeature(feature, layer) {
            layer.on({
                {%- if this.highlight %}
                mouseout: function(e) {
                    {{ this.get_name() }}.resetStyle(e.target);
                },
                mouseover: function(e) {
                    e.target.setStyle({{ this.get_name() }}_highlighter(e.target.feature));
                },
                {%- endif %}
            });
        };
        var {{ this.get_name() }} = L.geoJson(null, {
            {%- if this.smooth_factor is not none  %}
                smoothFactor: {{ this.smooth_factor|tojson }},
            {%- endif %}
                onEachFeature: {{ this.get_name() }}_onEachFeature,
            {% if this.style %}
                style: {{ this.get_name() }}_styler,
            {%- endif %}
        }).addTo({{ this._parent.get_name() }});
        {%- if this.embed %}
            {{ this.get_name() }}.addData({{ this.data|tojson }});
        {%- else %}
            $.ajax({url: {{ this.embed_link|tojson }}, dataType: 'json', async: true,
                success: function(data) {
                    {{ this.get_name() }}.addData(data);
            }});
        {%- endif %}
        {% endmacro %}
        """)  # noqa


class TimeSliderPolygonHeatmap(TimeSliderChoropleth):
    _template = Template(u"""
            {% macro script(this, kwargs) %}

                var timestamps = {{ this.timestamps }};
                var styledict = {{ this.styledict }};
                var current_timestamp = timestamps[0];

                // insert time polyheatmap_slider
                d3.select("body").insert("p", ":first-child").append("input")
                    .attr("type", "range")
                    .attr("width", "100px")
                    .attr("min", 0)
                    .attr("max", timestamps.length - 1)
                    .attr("value", 0)
                    .attr("id", "polyheatmap_slider")
                    .attr("step", "1")
                    .style('align', 'center');

                // insert time polyheatmap_slider output BEFORE time polyheatmap_slider (text on top of polyheatmap_slider)
                d3.select("body").insert("p", ":first-child").append("output")
                    .attr("width", "100")
                    .attr("id", "polyheatmap_slider-value")
                    .style('font-size', '18px')
                    .style('text-align', 'center')
                    .style('font-weight', '500%');

                var datestring = new Date(parseInt(current_timestamp)*1000).toISOString();
                d3.select("output#polyheatmap_slider-value").text(datestring);

                fill_map = function(){
                    for (var feature_id in styledict){
                        let style = styledict[feature_id]//[current_timestamp];
                        var fillColor = 'white';
                        var opacity = 0;
                        if (current_timestamp in style){
                            fillColor = style[current_timestamp]['color'];
                            opacity = style[current_timestamp]['opacity'];
                            d3.selectAll('#feature-'+feature_id)
                            .attr('fill', fillColor)
                            .style('fill-opacity', opacity);
                        }
                        d3.selectAll('#feature-'+feature_id)
                        .attr('stroke', 'white')
                        .attr('stroke-width', 0.5)
                        .attr('stroke-dasharray', '5,5')
                        .attr('fill-opacity', 0);
                    }
                }

                d3.select("#polyheatmap_slider").on("input", function() {
                    current_timestamp = timestamps[this.value];
                    var datestring = new Date(parseInt(current_timestamp)*1000).toISOString();
                    d3.select("output#polyheatmap_slider-value").text(datestring);
                fill_map();
                });

                {% if this.highlight %}
                    {{this.get_name()}}_onEachFeature = function onEachFeature(feature, layer) {
                        layer.on({
                            mouseout: function(e) {
                            if (current_timestamp in styledict[e.target.feature.id]){
                                var opacity = styledict[e.target.feature.id][current_timestamp]['opacity'];
                                d3.selectAll('#feature-'+e.target.feature.id).style('fill-opacity', opacity);
                            }
                        },
                            mouseover: function(e) {
                            if (current_timestamp in styledict[e.target.feature.id]){
                                d3.selectAll('#feature-'+e.target.feature.id).style('fill-opacity', 1);
                            }
                        },
                            click: function(e) {
                                {{this._parent.get_name()}}.fitBounds(e.target.getBounds());
                        }
                        });
                    };

                {% endif %}

                var {{this.get_name()}} = L.geoJson(
                    {% if this.embed %}{{this.style_data()}}{% else %}"{{this.data}}"{% endif %}
                    {% if this.smooth_factor is not none or this.highlight %}
                        , {
                        {% if this.smooth_factor is not none  %}
                            smoothFactor:{{this.smooth_factor}}
                        {% endif %}

                        {% if this.highlight %}
                            {% if this.smooth_factor is not none  %}
                            ,
                            {% endif %}
                            onEachFeature: {{this.get_name()}}_onEachFeature
                        {% endif %}
                        }
                    {% endif %}
                    ).addTo({{this._parent.get_name()}}
                );

            {{this.get_name()}}.setStyle(function(feature) {feature.properties.style;});

                {{ this.get_name() }}.eachLayer(function (layer) {
                    layer._path.id = 'feature-' + layer.feature.id;
                    });


                fill_map();

            {% endmacro %}
            """)

    def __init__(self, data, styledict, name=None, overlay=True, control=True,
                 show=True):
        super(TimeSliderPolygonHeatmap, self).__init__(data, styledict, name=name, overlay=overlay, control=control,
                                                       show=show)
        self.name = "TimeSliderPolygonHeatmap"


class TimeSliderDynamicMap(TimeSliderChoropleth):
        _template = Template(u"""
                {% macro script(this, kwargs) %}

                    var timestamps = {{ this.timestamps }};
                    var styledict = {{ this.styledict }};
                    var current_timestamp = timestamps[0];

                    // insert time dynmap_slider
                    d3.select("body").insert("p", ":first-child").append("input")
                        .attr("type", "range")
                        .attr("width", "100px")
                        .attr("min", 0)
                        .attr("max", timestamps.length - 1)
                        .attr("value", 0)
                        .attr("id", "dynmap_slider")
                        .attr("step", "1")
                        .style('align', 'center');

                    // insert time dynmap_slider output BEFORE time dynmap_slider (text on top of dynmap_slider)
                    d3.select("body").insert("p", ":first-child").append("output")
                        .attr("width", "100")
                        .attr("id", "dynmap_slider-value")
                        .style('font-size', '18px')
                        .style('text-align', 'center')
                        .style('font-weight', '500%');

                    var datestring = new Date(parseInt(current_timestamp)*1000).toISOString();
                    d3.select("output#dynmap_slider-value").text(datestring);

                    fill_map = function(){
                        for (var feature_id in styledict){
                            let style = styledict[feature_id]//[current_timestamp];
                            var fillColor = 'white';
                            var opacity = 0;
                            if (current_timestamp in style){
                                fillColor = style[current_timestamp]['color'];
                                opacity = style[current_timestamp]['opacity'];
                                d3.selectAll('#feature-'+feature_id)
                                .attr('fill', fillColor)
                                .style('fill-opacity', opacity);
                            }
                            d3.selectAll('#feature-'+feature_id)
                            .attr('stroke', 'white')
                            .attr('stroke-width', 0.0)
                            .attr('stroke-dasharray', '5,5')
                            .attr('fill-opacity', 0);
                        }
                    }


                    d3.select("#dynmap_slider").on("input", function() {
                        current_timestamp = timestamps[this.value];
                        var datestring = new Date(parseInt(current_timestamp)*1000).toISOString();
                        d3.select("output#dynmap_slider-value").text(datestring);
                        fill_map();
                    });

                    {% if this.highlight %}
                        {{this.get_name()}}_onEachFeature = function onEachFeature(feature, layer) {
                            layer.on({
                                mouseout: function(e) {
                                if (current_timestamp in styledict[e.target.feature.id]){
                                    var opacity = styledict[e.target.feature.id][current_timestamp]['opacity'];
                                    d3.selectAll('#feature-'+e.target.feature.id).style('fill-opacity', opacity);
                                }
                            },
                                mouseover: function(e) {
                                if (current_timestamp in styledict[e.target.feature.id]){
                                    d3.selectAll('#feature-'+e.target.feature.id).style('fill-opacity', 1);
                                }
                            },
                                click: function(e) {
                                    {{this._parent.get_name()}}.fitBounds(e.target.getBounds());
                            }
                            });
                        };

                    {% endif %}

                    var {{this.get_name()}} = L.geoJson(
                        {% if this.embed %}{{this.style_data()}}{% else %}"{{this.data}}"{% endif %}
                        {% if this.smooth_factor is not none or this.highlight %}
                            , {
                            {% if this.smooth_factor is not none  %}
                                smoothFactor:{{this.smooth_factor}}
                            {% endif %}

                            {% if this.highlight %}
                                {% if this.smooth_factor is not none  %}
                                ,
                                {% endif %}
                                onEachFeature: {{this.get_name()}}_onEachFeature
                            {% endif %}
                            }
                        {% endif %}
                        ).addTo({{this._parent.get_name()}}
                    );

                {{this.get_name()}}.setStyle(function(feature) {feature.properties.style;});

                    {{ this.get_name() }}.eachLayer(function (layer) {
                        layer._path.id = 'feature-' + layer.feature.id;
                        });


                    fill_map();

                {% endmacro %}
                """)

        def __init__(self, data, styledict, name=None, overlay=True, control=True,
                     show=True):
            super(TimeSliderDynamicMap, self).__init__(data, styledict, name=name, overlay=overlay, control=control,
                                                       show=show)
            self.name = "TimeSliderDynamicMap"


class LatLngCopy(MacroElement):
    """
    When one clicks on a Map, it copies the latitude and longitude of the pointer to self.loc.
    """
    _template = Template(u"""
            {% macro script(this, kwargs) %}
                function latLngCopy(e) {
                    {{this._parent.get_name()}}.loc = [e.latlng.lat, e.latlng.lng]
                }
                {{this._parent.get_name()}}.on('click', latLngCopy);
                {{this._parent.get_name()}}.loc = [0, 0]
            {% endmacro %}
            """)  # noqa

    def __init__(self):
        super().__init__()
        self._name = 'LatLgCopy'

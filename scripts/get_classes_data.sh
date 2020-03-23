#!/usr/bin/env bash

EARTH_LOCATION=$1

QUERY_PEAKS='node[natural=peak]'${EARTH_LOCATION}';(._;>;);out;'
QUERY_STREAMS='way[waterway=stream]'${EARTH_LOCATION}';(._;>;);out;'
QUERY_RIVERS='way[waterway=river]'${EARTH_LOCATION}';(._;>;);out;'
QUERY_CLIFFS='way[natural=cliff]'${EARTH_LOCATION}';(._;>;);out;'

declare -a INDEXES=(0 1 2 3)
declare -a QUERIES=('node[natural=peak]'${EARTH_LOCATION}';(._;>;);out;'
 'way[waterway=stream]'${EARTH_LOCATION}';(._;>;);out;'
 'way[waterway=river]'${EARTH_LOCATION}';(._;>;);out;'
 'way[natural=cliff]'${EARTH_LOCATION}';(._;>;);out;'
);
declare -a NAMES=("peaks" "streams" "rivers" "cliffs")

for i in "${INDEXES[@]}"
do
    SAVE_LOCATION='/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/overpass_classes_data/'${NAMES[i]}'_'${EARTH_LOCATION}'.geojson'
    echo ${SAVE_LOCATION}
    LINK='http://overpass-api.de/api/interpreter?data=[out:json];'${QUERIES[i]}
    echo ${LINK}
    wget -O ${SAVE_LOCATION} --no-check-certificate ${LINK}
done

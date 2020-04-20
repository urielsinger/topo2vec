#!/usr/bin/env bash

EARTH_LOCATION=$1

QUERY_PEAKS='node[natural=peak]'${EARTH_LOCATION}';(._;>;);out;'
QUERY_STREAMS='way[waterway=stream]'${EARTH_LOCATION}';(._;>;);out;'
QUERY_RIVERS='way[waterway=river]'${EARTH_LOCATION}';(._;>;);out;'
QUERY_CLIFFS='way[natural=cliff]'${EARTH_LOCATION}';(._;>;);out;'

declare -a INDEXES=(0 1 2 3)
declare -a QUERIES=('way[natural=cliff]'${EARTH_LOCATION}';(._;>;);out;'
 'node[natural=peak]'${EARTH_LOCATION}';(._;>;);out;'
 'way[waterway=stream]'${EARTH_LOCATION}';(._;>;);out;'
 'way[waterway=river]'${EARTH_LOCATION}';(._;>;);out;'
);

QUERY_SADDLES='node[natural=saddle]'${EARTH_LOCATION}';(._;>;);out;'
QUERY_ROCKS='node[natural=rock]'${EARTH_LOCATION}';(._;>;);out;'
QUERY_CAVE_ENTRANCE='node[natural=cave_entrance]'${EARTH_LOCATION}';(._;>;);out;'
QUERY_SINK_HOLE='node[natural=sinkhole]'${EARTH_LOCATION}';(._;>;);out;'


declare -a QUERIES2=('node[natural=saddle]'${EARTH_LOCATION}';(._;>;);out;'
 'node[natural=rock]'${EARTH_LOCATION}';(._;>;);out;'
 'node[natural=cave_entrance]'${EARTH_LOCATION}';(._;>;);out;'
 'node[natural=sinkhole]'${EARTH_LOCATION}';(._;>;);out;'
);

declare -a QUERIES3=('node[natural=ridge]'${EARTH_LOCATION}';(._;>;);out;'
 'node[aerialway=station]'${EARTH_LOCATION}';(._;>;);out;'
 'node[natural=volcano]'${EARTH_LOCATION}';(._;>;);out;'
 'node[power=tower]'${EARTH_LOCATION}';(._;>;);out;'
);

declare -a QUERIES4=('node[man_made=antenna]'${EARTH_LOCATION}';(._;>;);out;'
 'node[man_made=communications_tower]'${EARTH_LOCATION}';(._;>;);out;'
 'node[waterway=waterfall]'${EARTH_LOCATION}';(._;>;);out;'
 'node[tourism=alpine_hut]'${EARTH_LOCATION}';(._;>;);out;'
);

declare -a NAMES=("cliffs" "peaks" "streams" "rivers")
declare -a NAMES2=("saddles" "rocks" "cave_entrances" "sinkholes")
declare -a NAMES3=("ridges"  "airialway_stations" "volcanos" "power_towers")
declare -a NAMES4=("antenas"  "communications_towers" "waterfalls" "alpine_huts")

for i in "${INDEXES[@]}"
do
    SAVE_LOCATION='//home/topo2vec_kavitzky/topo2vec/data/overpass_classes_data/'${NAMES3[i]}'_'${EARTH_LOCATION}'.json'
    echo ${SAVE_LOCATION}
    LINK='http://overpass-api.de/api/interpreter?data=[out:json];'${QUERIES3[i]}
    echo ${LINK}
    wget -O ${SAVE_LOCATION} --no-check-certificate ${LINK}
done

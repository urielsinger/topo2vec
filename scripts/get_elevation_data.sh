#!/usr/bin/env bash
# usage: the first input e.g. N035E035_N040E040 for israel's area
# the data is originally from the site: https://www.eorc.jaxa.jp/ALOS/en/aw3d30/
#username: googooodle@gmail.com
#password: aw3d30
EARTH_LOCATION=$1
SAVE_LOCATION='.'${EARTH_LOCATION}'.zip'
LINK='133.56.96.210/ALOS/aw3d30/data/release_v2003//'${EARTH_LOCATION}'.zip'
EXTRACT_LOCATION='/home/topo2vec_kavitzky/topo2vec/data/elevation'
wget -O ${SAVE_LOCATION} --http-user=googooodle@gmail.com --http-password=aw3d30 --no-check-certificate ${LINK}
unzip ${SAVE_LOCATION} -d ${EXTRACT_LOCATION}
#tar xvzf ${SAVE_LOCATION} -C ${EXTRACT_LOCATION} --strip 1
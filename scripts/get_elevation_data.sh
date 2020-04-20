#!/usr/bin/env bash
EARTH_LOCATION=$1
SAVE_LOCATION='.'${EARTH_LOCATION}'.tar.gz'
LINK='133.56.96.210/ALOS/aw3d30/data/release_v1903/'${EARTH_LOCATION}'.tar.gz'
EXTRACT_LOCATION='/home/topo2vec_kavitzky/topo2vec/data/elevation'

wget -O ${SAVE_LOCATION} --http-user=googooodle@gmail.com --http-password=aw3d30 --no-check-certificate ${LINK}
tar xvzf ${SAVE_LOCATION} -C ${EXTRACT_LOCATION} --strip 1
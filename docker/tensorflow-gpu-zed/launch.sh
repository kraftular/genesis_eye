#!/bin/bash

#note: the --privileged arg is dangerous, but it's the only hope of connecting a camera
#to the docker container, it seems. if you don't need to use a camera (processing saved
#videos, for example) you can remove the --privileged arg to close the gaping security
#hole it rips open.
#
#the --net=host makes it look like the container is your host as far as X forwarding is
#concerned, so that you can e.g. ssh -Y into a machine running docker and see the X gui
#windows created by the container.
#
#you need to pass an argument to this script: the location of the host directory that
#will be mapped as /home/genesis/genesis_eye on the container. this directory should
#probably be my git repository, but it can be anything you want to map to the container
#
#REMEMBER THAT ANY WORK SAVED OUTSIDE OF /home/genesis/genesis_eye will be blown away
#when the container exits. if that's not the behavior you want, you can remove the --rm
#option


GENESIS_EYE_DIR=$1

if [ ! -d "${GENESIS_EYE_DIR}" ]; then
    echo "you need to supply the name of a directory to map to /home/genesis/genesis_eye"
    exit 1
fi

docker run --gpus all -it --rm --privileged  --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/home/genesis/.Xauthority:rw" -v ${GENESIS_EYE_DIR}:/home/genesis/genesis_eye genesis/tfzed:tf_2.3gpu_zed_3.2-devel-cuda11.0-ubuntu18.04

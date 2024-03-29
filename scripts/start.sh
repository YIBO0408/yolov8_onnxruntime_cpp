#!/bin/bash

parent_dir=$(dirname "$(pwd)")

docker run \
--privileged \
--gpus all \
--rm \
-it \
-v $parent_dir:$parent_dir \
-w $parent_dir/scripts \
hub.micro-i.com.cn:9443/dad/tritonserver_dev:23.09-py3 bash

# !/bin/bash 

xhost local:root

sudo nvidia-docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --device /dev/video0 -v /home/deeplift/container_share/:/container_share -p 8008:8008 -it 08d13d6e7420 /bin/bash

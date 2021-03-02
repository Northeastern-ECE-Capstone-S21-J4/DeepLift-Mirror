# !/bin/bash 

sudo nvidia-docker run --device /dev/video0 -v /home/deeplift/container_share/:/container_share -p 8888:8888 -it 7ee7bd70fce1 /bin/bash

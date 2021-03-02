# !/bin/bash

nvidia-docker run --device /dev/video0 -v /home/deeplift/container_share/:/container_share -p 8888:8888 -it 7ee7bd70fce1 jupyter notebook /container_share/trt_pose/tasks/human_pose/live_demo.ipynb --allow-root --ip 0.0.0.0

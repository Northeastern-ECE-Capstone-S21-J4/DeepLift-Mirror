This is the source control for pose estimation on the Jetson Xavier, located in
the /container_share folder of the docker image "deeplift-capstone/mirror-fw" which
mirrors the ~/container_share folder of the jetson filesystem.

The docker container can be started by running the ./start_container.sh script.

The main demo is located in ./trt_pose/tasks/human_pose/live_demo.ipynb. This should be 
run from outside the container by running ~/run_trt_pose.sh which will open up a jupyter notebook
in the web browser. 

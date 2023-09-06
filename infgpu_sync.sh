while true; do
	sleep 3
	rsync --exclude home/wolter/uni/diffusion-data -r -a -v /home/wolter/uni/diffusion wolter@lmgpu-login.informatik.uni-bonn.de:/home/wolter/uni 
done
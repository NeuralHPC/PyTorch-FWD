while true; do
	sleep 5
	rsync -r -a -v /home/wolter/uni/diffusion wolter@lmgpu-login.informatik.uni-bonn.de:/home/wolter/uni
done
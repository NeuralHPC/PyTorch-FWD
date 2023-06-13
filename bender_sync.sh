while true; do
	sleep 5
	rsync -r -a -v /home/wolter/uni/diffusion mwolter1@bender.hpc.uni-bonn.de:/home/mwolter1/uni/
done
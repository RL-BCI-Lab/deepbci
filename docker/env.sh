#!bin/bash
# Resets OPTIND so env can be sourced multiple times
OPTIND=1 
user="1000:10"
version="tf"
path=$PWD

while getopts u:v:p: option
do
case "${option}"
in
u) user=${OPTARG};;
v) version=${OPTARG};;
p) path=${OPTARG};;

esac
done

echo $version $user $path

alias dbci-make="docker run \
    --volume=$path:/home/dev/mnt \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix \
    --volume=/etc/localtime:/etc/localtime:ro \
    --name dbci-$version  \
    --user $user \
    --shm-size 16G \
    -dit \
    -e DISPLAY \
    -e XAUTHORITY \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 8866:8866 \
    -p 8000:8000 \
    bpoole908/deep-bci:$version /bin/bash"

alias dbci-attach="docker run \
    --volume=$path:/home/dev/mnt \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix \
    --volume=/etc/localtime:/etc/localtime:ro \
    --rm \
    --name dbci-$version-tmp \
    --user $user \
    --shm-size 16G \
    -dit \
    -e DISPLAY \
    -e XAUTHORITY \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 8866:8866 \
    -p 8000:8000 \
    bpoole908/deep-bci:$version /bin/bash \
    && docker attach dbci-$version-tmp"

alias dbci-gpu-make="docker run \
    --volume=$path:/home/dev/mnt \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix \
    --volume=/etc/localtime:/etc/localtime:ro \
    --name dbci-gpu-$version  \
    --user $user \
    --gpus all \
    --shm-size 16G \
    -dit \
    -e DISPLAY \
    -e XAUTHORITY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 8866:8866 \
    -p 8000:8000 \
    bpoole908/deep-bci:$version /bin/bash"

alias dbci-gpu-attach="docker run \
    --volume=$path:/home/dev/mnt \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix \
    --volume=/etc/localtime:/etc/localtime:ro \
    --rm \
    --name dbci-gpu-$version-tmp \
    --user $user \
    --gpus all \
    --shm-size 16G \
    -dit \
    -e DISPLAY \
    -e XAUTHORITY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 8866:8866 \
    -p 8000:8000 \
    bpoole908/deep-bci:$version /bin/bash \
    && docker attach dbci-gpu-$version-tmp"

alias dbci-wsl-make="docker run \
    --volume=$path:/home/dev/mnt \
    --volume=/etc/localtime:/etc/localtime:ro \
    --name dbci-wsl-$version  \
    --user $user \
    --shm-size 16G \
    -dit \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 8866:8866 \
    -p 8000:8000 \
    bpoole908/deep-bci:$version /bin/bash "

alias dbci-wsl-attach="docker run \
    --volume=$path:/home/dev/mnt \
    --volume=/etc/localtime:/etc/localtime:ro \
    --rm \
    --name dbci-wsl-$version-tmp \
    --user $user \
    --shm-size 16G \
    -dit \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 8866:8866 \
    -p 8000:8000 \
    bpoole908/deep-bci:$version /bin/bash
    & docker attach dbci-wsl-$version-tmp"

alias dbci-wsl-gpu-make="docker run \
    --volume=$path:/home/dev/mnt \
    --volume=/etc/localtime:/etc/localtime:ro \
    --name dbci-wsl-gpu-$version  \
    --user $user \
    --gpus all \
    --shm-size 16G \
    -dit \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 8866:8866 \
    -p 8000:8000 \
    bpoole908/deep-bci:$version /bin/bash"

alias dbci-wsl-gpu-attach="docker run \
    --volume=$path:/home/dev/mnt \
    --volume=/etc/localtime:/etc/localtime:ro \
    --rm \
    --name dbci-wsl-gpu-$version-tmp  \
    --user $user \
    --gpus all \
    --shm-size 16G \
    -dit \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 8866:8866 \
    -p 8000:8000 \
    bpoole908/deep-bci:$version /bin/bash
    & docker attach dbci-wsl-gpu-$version-tmp"

#!bin/bash
# Resets OPTIND so env can be sourced multiple times
OPTIND=1 
partition="Hercules"
nodes="1"
mem_per_cpu="4"
gpus="GTX1080ti:8"
cpus="8"

while getopts p:n:m:g: flag; do
    case "${flag}" in
        p) partition=${OPTARG};;
        n) nodes=${OPTARG};;
        m) mem_per_cpu=${OPTARG};;
        g) gpus=${OPTARG};;
        c) cpus=${OPTARG};;
        *) echo "Invalid option: -$flag" ;;
    esac
done

echo $partition $nodes $mem_per_cpu $gpus $cpus
export GPUS=${gpus: -1}
alias make-interactive="srun \
    --mpi=pmix \
    --partition=${partition} \
    --nodes=${nodes} \
    --mem-per-cpu=${mem_per_cpu}GB \
    --gres=gpu:${gpus} \
    --cpus-per-task=${cpus} \
    --time=30:00:00 \
    --job-name=intractive-session \
    --pty /bin/bash"

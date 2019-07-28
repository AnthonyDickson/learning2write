#!/usr/bin/env bash

n_steps=1000000

usage="$(basename "$0") [-h] [-n n] -- Utility for running experiments.
where:
    -h          Show this help text and exit.
    -n N_STEPS  How many steps to train each agent for (default: ${n_steps})"


while getopts ':hn:' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    n) n_steps=$OPTARG
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

x_server_num=0
mkdir -p logs

function start_run() {
    x_server_num=$((x_server_num+1))

    nohup xvfb-run -e /dev/stdout -s "-screen 0 1200x800x24" -n ${x_server_num} \
 	python train.py -model-type $1 -policy-type $2 -pattern-set $3 -updates ${n_steps} \
	&> logs/nohup.$1_$2_$3.out &

}

start_run acktr mlp 5x5
start_run ppo mlp 5x5

start_run acktr mlp digits
start_run ppo mlp digits

start_run acktr mlp emnist
start_run ppo mlp emnist

start_run acktr cnn emnist
start_run ppo cnn emnist


nohup tensorboard --logdir tensorboard/ &> logs/nohup.tensorboard.out &



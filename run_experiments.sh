
nohup xvfb-run -s "-screen 0 1200x800x24" -n=1 python train.py -model-type=acktr -pattern-set 5x5 -updates 100000000 &> nohup.acktr_5x5.out &
nohup xvfb-run -s "-screen 0 1200x800x24" -n=2 python train.py -model-type=ppo -pattern-set 5x5 -updates 100000000 &> nohup.ppo_5x5.out &

nohup xvfb-run -s "-screen 0 1200x800x24" -n=3 python train.py -model-type=acktr -pattern-set digits -updates 100000000 &> nohup.acktr_digits.out &
nohup xvfb-run -s "-screen 0 1200x800x24" -n=4 python train.py -model-type=ppo -pattern-set digits -updates 100000000 &> nohup.ppo_digits.out &
nohup tensorboard --logdir tensorboard/ &> nohup.tensorboard.out &

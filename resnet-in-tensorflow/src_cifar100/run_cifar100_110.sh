source activate deep-learning

echo "################################## exp11:Start: lam=1000 net=110 ##########################################"
python cifar10_train.py --version=lr10m4_lam10e3_cifar100_resnet110_decayevery10k --gpu='3' --init_lr=0.0001 --lam=1000 --num_residual_blocks=18
echo "################################## exp11:End: lam=1000 net=110 ##########################################"

echo "################################## exp12:Start: lam=100 net=110 ##########################################"
python cifar10_train.py --version=lr10m4_lam10e2_cifar100_resnet110_decayevery10k --gpu='3' --init_lr=0.0001 --lam=100 --num_residual_blocks=18
echo "################################## exp12:End: lam=100 net=110 ##########################################"




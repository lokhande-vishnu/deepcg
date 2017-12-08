source activate deep-learning

echo "################################## exp1:Start: lam=1000 net=32 ##########################################"
python cifar10_train.py --version=lr10m4_lam10e3_cifar10_resnet32_decayevery10k --gpu='0' --init_lr=0.0001 --lam=1000 --num_residual_blocks=5
echo "################################## exp1:End: lam=1000 net=32 ##########################################"

echo "################################## exp2:Start: lam=1000 net=56 ##########################################"
python cifar10_train.py --version=lr10m4_lam10e3_cifar10_resnet56_decayevery10k --gpu='0' --init_lr=0.0001 --lam=1000 --num_residual_blocks=9
echo "################################## exp2:End: lam=1000 net=56 ##########################################"

echo "################################## exp3:Start: lam=100 net=32 ##########################################"
python cifar10_train.py --version=lr10m4_lam10e2_cifar10_resnet32_decayevery10k --gpu='0' --init_lr=0.0001 --lam=100 --num_residual_blocks=5
echo "################################## exp3:End: lam=100 net=32 ##########################################"

echo "################################## exp4:Start: lam=100 net=56 ##########################################"
python cifar10_train.py --version=lr10m4_lam10e2_cifar10_resnet56_decayevery10k --gpu='0' --init_lr=0.0001 --lam=100 --num_residual_blocks=9
echo "################################## exp1:End: lam=100 net=56 ##########################################"



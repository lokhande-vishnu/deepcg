source activate deep-learning

echo "################################## exp7:Start: lam=1000 net=32 ##########################################"
python cifar10_train.py --version=lr10m4_lam10e3_cifar100_resnet32_decayevery10k --gpu='2' --init_lr=0.0001 --lam=1000 --num_residual_blocks=5
echo "################################## exp7:End: lam=1000 net=32 ##########################################"

echo "################################## exp8:Start: lam=1000 net=56 ##########################################"
python cifar10_train.py --version=lr10m4_lam10e3_cifar100_resnet56_decayevery10k --gpu='2' --init_lr=0.0001 --lam=1000 --num_residual_blocks=9
echo "################################## exp8:End: lam=1000 net=56 ##########################################"

echo "################################## exp9:Start: lam=100 net=32 ##########################################"
python cifar10_train.py --version=lr10m4_lam10e2_cifar100_resnet32_decayevery10k --gpu='2' --init_lr=0.0001 --lam=100 --num_residual_blocks=5
echo "################################## exp9:End: lam=100 net=32 ##########################################"

echo "################################## exp10:Start: lam=100 net=56 ##########################################"
python cifar10_train.py --version=lr10m4_lam10e2_cifar100_resnet56_decayevery10k --gpu='2' --init_lr=0.0001 --lam=100 --num_residual_blocks=9
echo "################################## exp10:End: lam=100 net=56 ##########################################"



echo "run_testlfw_varyepochs"

for i in {0..25..5}
do
    echo "lamda10e5_traincelebA $i"
    ./complete.py ../lfw_aligned/lfw_new/* --checkpointDir lamda10e5_traincelebA/EPOCH-${i} --outDir lamda10e5_traincelebA/EPOCH-${i} 
done


for i in {0..25..5}
do
    echo "lamda10e6_traincelebA $i"
    ./complete.py ../lfw_aligned/lfw_new/* --checkpointDir lamda10e6_traincelebA/EPOCH-${i} --outDir lamda10e6_traincelebA/EPOCH-${i} 
done

for i in {0..25..5}
do
    echo "lamda10_traincelebA $i"
    ./complete.py ../lfw_aligned/lfw_new/* --checkpointDir lamda10_traincelebA/EPOCH-${i} --outDir lamda10_traincelebA/EPOCH-${i} 
done

    
for i in {0..25..5}
do
    echo "lamda10e3_traincelebA $i"
    ./complete.py ../lfw_aligned/lfw_new/* --checkpointDir lamda10e3_traincelebA/EPOCH-${i} --outDir lamda10e3_traincelebA/EPOCH-${i} 
done
      

#!/bin/bash
source activate tensorflow
for k in $( seq 9  20)
do
   python cnn_res.py --model_path='cnn_res/models/'${k} --pretrain='0'
done

for k in $(seq 21 30)
do
   python cnn_res.py --model_path='cnn_res/models/'${k} --pretrain='1'
done

cd ../attmfl_c3d
python feature_extract.py

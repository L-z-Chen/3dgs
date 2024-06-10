#!/bin/bash

# Constants for learning rates
position_lr_init=0.00009
feature_lr=0.0004
position_lr_final=0.0000009
gradmx=0.0002

base_ip="127.0.1."
ip_start=5
lambda_dssim=0.2
opacity_lr=0.03
# path="./data/tandt/train"
path="/scratch/cluster/lzchen/nerf360/bicycle"

# Extract the last word of the path
data_name=$(basename $path)

# Ensure the logs directory exists
mkdir -p ./logs  

# Always use GPU 0
current_gpu=2

# Define the base IP address and index
index=4
total_gpus=8

# Loop over lambda_dssim values
for opacity_lr in "${opacity_lr[@]}"
do
for lambda_val in "${lambda_dssim[@]}"
do  
    model_path="./output/${data_name}_lrinit${position_lr_init}_ftrlr${feature_lr}_gradmx${gradmx}_lamb${lambda_dssim}_oplr${opacity_lr}"
    current_gpu=$((index % total_gpus))
    current_ip="${base_ip}$((ip_start + index))"
    output_file="./logs/${data_name}_gpu${current_gpu}_poslr${position_lr_init}_featlr${feature_lr}_gradmx${gradmx}_lambda${lambda_val}_opacity_lr${opacity_lr}_ip${current_ip}.log"
    
    echo $output_file, $model_path
    echo "Training with GPU=$current_gpu, position_lr_init=$position_lr_init, feature_lr=$feature_lr, lambda_dssim=$lambda_val, IP=$current_ip"
    CUDA_VISIBLE_DEVICES=$current_gpu python train.py -s $path \
    --ip $current_ip --use_sh 'hg' --position_lr_init $position_lr_init --feature_lr $feature_lr --position_lr_final $position_lr_final \
    --densify_grad_threshold $gradmx \
    --lambda_dssim $lambda_val \
    --opacity_lr $opacity_lr \
    --model_path $model_path \
    > $output_file 2>&1 &
    
    
    # Wait for the background job to finish
    # wait

    # Wait for 1.2 hours (4320 seconds) before starting the next run
    echo "Waiting for 1.2 hours before starting the next training session..."
    # sleep 4320

    index=$((index + 1))
done
done
echo "All training jobs have completed."

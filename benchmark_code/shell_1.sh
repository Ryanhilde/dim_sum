#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
# Define the models as a space-separated string
models="Autoformer BRITS Crossformer CSDI DLinear ETSformer"

# Loop through each model and run the python script
for model in $models
do
    echo "Running model: $model"
    python federated_train_model.py \
        --model "$model" \
        --dataset Electricity \
        --dataset_fold_path ./data/generated_datasets/electricity_load_diagrams_rate01_step96_point \
        --saving_path ./results \
        --device cuda:0 \
        --n_clients 5 \
        --global_rounds 100 \
        --local_epochs 1
    
    rm -rf "./results/${model}_Electricity_federated"
    echo "Deleted results for ${model}"
done

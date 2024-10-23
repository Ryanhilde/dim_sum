"""
Federated Learning implementation for time series imputation models
"""

# Created by AI Assistant
# License: BSD-3-Clause

import argparse
import os
import time
import copy
import numpy as np
import csv
import torch
from pypots.data.saving import pickle_dump
from pypots.imputation import *
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre
from pypots.utils.random import set_random_seed

from global_config import (
    TORCH_N_THREADS,
    RANDOM_SEEDS,
)
from hpo_results import HPO_RESULTS
from utils import get_datasets_path

SUPPORT_MODELS = {
    "Autoformer": Autoformer,
    "BRITS": BRITS,
    "Crossformer": Crossformer,
    "CSDI": CSDI,
    "DLinear": DLinear,
    "ETSformer": ETSformer,
    "FiLM": FiLM,
    "FreTS": FreTS,
    "GPVAE": GPVAE,
    "GRUD": GRUD,
    "Informer": Informer,
    "iTransformer": iTransformer,
    "Koopa": Koopa,
    "MICN": MICN,
    "MRNN": MRNN,
    "NonstationaryTransformer": NonstationaryTransformer,
    "PatchTST": PatchTST,
    "Pyraformer": Pyraformer,
    "SAITS": SAITS,
    "SCINet": SCINet,
    "StemGNN": StemGNN,
    "TimesNet": TimesNet,
    "Transformer": Transformer,
    "USGAN": USGAN,
}
SUPPORT_DATASETS = list(HPO_RESULTS.keys())

class FederatedModel:
    def __init__(self, model_class, model_params):
        self.model_class = model_class
        self.model_params = model_params
        self.model = self.model_class(**self.model_params)

    def fit(self, train_set, val_set, epochs):
        self.model.fit(train_set=train_set, val_set=val_set)

    def predict(self, test_set):
        return self.model.predict(test_set)

    def state_dict(self):
        return self.model.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.model.load_state_dict(state_dict)

def aggregate_models(global_model, local_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([local_model.state_dict()[k].float() for local_model in local_models], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def federated_train(model_class, model_params, clients_data, global_rounds, local_epochs, device):
    global_model = FederatedModel(model_class, model_params)
    
    for round in range(global_rounds):
        logger.info(f"Global Round {round + 1}/{global_rounds}")
        
        local_models = []
        for client_id, client_data in enumerate(clients_data):
            logger.info(f"Training on Client {client_id + 1}/{len(clients_data)}")
            local_model = FederatedModel(model_class, model_params)
            local_model.load_state_dict(global_model.state_dict())
            
            local_model.fit(train_set=client_data['train'], val_set=client_data['val'], epochs=local_epochs)
            local_models.append(local_model)
        
        # Aggregate models
        global_model = aggregate_models(global_model, local_models)
    
    return global_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="the model name",
        required=True,
        choices=list(SUPPORT_MODELS.keys()),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="the dataset name",
        required=True,
        choices=SUPPORT_DATASETS,
    )
    parser.add_argument(
        "--dataset_fold_path",
        type=str,
        help="the dataset fold path, where should include 3 H5 files train.h5, val.h5 and test.h5",
        required=True,
    )
    parser.add_argument(
        "--saving_path",
        type=str,
        help="the saving path of the model and logs",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device to run the model, e.g. cuda:0",
        required=True,
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        help="the number of rounds running the model to get the final results ",
        default=1,
    )
    parser.add_argument(
        "--n_clients",
        type=int,
        help="the number of clients for federated learning",
        default=5,
    )
    parser.add_argument(
        "--global_rounds",
        type=int,
        help="the number of global rounds for federated learning",
        default=10,
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        help="the number of local epochs for each client",
        default=1,
    )
    args = parser.parse_args()

    # Set the number of threads for PyTorch
    torch.set_num_threads(TORCH_N_THREADS)
    
    # Load and split data for federated learning
    #print(args.dataset_fold_path)
    train_set, val_set, test_X, test_X_ori, test_indicating_mask = get_datasets_path(args.dataset_fold_path)

    # Split data into clients (modified to handle different sizes of train and val sets)
    clients_data = []
    n_train_samples = len(train_set['X'])
    n_val_samples = len(val_set['X'])

    train_indices = np.random.permutation(n_train_samples)
    val_indices = np.random.permutation(n_val_samples)

    train_split_indices = np.array_split(train_indices, args.n_clients)
    val_split_indices = np.array_split(val_indices, args.n_clients)

    for train_client_indices, val_client_indices in zip(train_split_indices, val_split_indices):
        client_train = {k: v[train_client_indices] for k, v in train_set.items()}
        client_val = {k: v[val_client_indices] for k, v in val_set.items()}
        clients_data.append({
            'train': client_train,
            'val': client_val
        })

    mae_collector = []
    mse_collector = []
    mre_collector = []
    time_collector = []

    result_saving_path = os.path.join(args.saving_path, f"{args.model}_{args.dataset}_federated")
    start_time = time.time()
    for n_round in range(args.n_rounds):
        set_random_seed(RANDOM_SEEDS[n_round])
        round_saving_path = os.path.join(result_saving_path, f"round_{n_round}")

        # Get the hyperparameters and setup the model
        hyperparameters = HPO_RESULTS[args.dataset][args.model].copy()
        lr = hyperparameters.pop("lr")
        hyperparameters["device"] = args.device
        hyperparameters["saving_path"] = round_saving_path
        hyperparameters["model_saving_strategy"] = "best"
        hyperparameters["epochs"] = args.local_epochs
        hyperparameters["patience"] = 1
        if args.model == "USGAN":
            hyperparameters["G_optimizer"] = Adam(lr=lr)
            hyperparameters["D_optimizer"] = Adam(lr=lr)
        else:
            hyperparameters["optimizer"] = Adam(lr=lr)

        model_class = SUPPORT_MODELS[args.model]
        federated_model = federated_train(model_class, hyperparameters, clients_data, args.global_rounds, args.local_epochs, args.device)
        time_collector.append(time.time() - start_time)

        # Evaluate the federated model
        test_set = {"X": test_X}
        results = federated_model.predict(test_set)
        test_set_imputation = results["imputation"]

        mae = calc_mae(test_set_imputation, test_X_ori, test_indicating_mask)
        mse = calc_mse(test_set_imputation, test_X_ori, test_indicating_mask)
        mre = calc_mre(test_set_imputation, test_X_ori, test_indicating_mask)
        mae_collector.append(mae)
        mse_collector.append(mse)
        mre_collector.append(mre)

        pickle_dump(
            {
                "test_set_imputation": test_set_imputation,
            },
            os.path.join(round_saving_path, "imputation.pkl"),
        )
        logger.info(
            f"Round{n_round} - Federated {args.model} on {args.dataset}: MAE={mae:.4f}, MSE={mse:.4f}, MRE={mre:.4f}"
        )

    mean_mae, mean_mse, mean_mre = np.mean(mae_collector), np.mean(mse_collector), np.mean(mre_collector)
    std_mae, std_mse, std_mre = np.std(mae_collector), np.std(mse_collector), np.std(mre_collector)
    num_params = sum(p.numel() for p in federated_model.model.model.parameters() if p.requires_grad)
    logger.info(
        f"Done! Final results:\n"
        f"Averaged Federated {args.model} ({num_params:,} params) on {args.dataset}: "
        f"MAE={mean_mae:.4f} ± {std_mae}, "
        f"MSE={mean_mse:.4f} ± {std_mse}, "
        f"MRE={mean_mre:.4f} ± {std_mre}, "
        f"average training time={np.mean(time_collector):.2f}"
    )

        # Define the CSV file path
    csv_file_path = os.path.join(args.saving_path, "federated_results.csv")

    # Prepare the data to be written
    data = {
        "Model": args.model,
        "Dataset": args.dataset,
        "Num_Params": num_params,
        "MAE_Mean": mean_mae,
        "MAE_Std": std_mae,
        "MSE_Mean": mean_mse,
        "MSE_Std": std_mse,
        "MRE_Mean": mean_mre,
        "MRE_Std": std_mre,
        "Avg_Training_Time": np.mean(time_collector)
    }

    # Check if the file exists to determine if we need to write headers
    file_exists = os.path.isfile("federated_results.csv")

    # Open the CSV file in append mode
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = list(data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write headers if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Write the data
        writer.writerow(data)

    logger.info(f"Results appended to {csv_file_path}")

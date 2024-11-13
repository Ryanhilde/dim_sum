import os
import pickle
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre
from utils import get_datasets_path
from pypots.utils.logging import logger

# Define the datasets
DATASET_PATHS = [
    "ett_rate01_step48_point",
    "ett_rate03_step48_block_blocklen6",
    "ett_rate05_step48_subseq_seqlen36",
    "ett_rate05_step48_point",
    "ett_rate09_step48_point"
]

# Define the models
MODELS = ["Autoformer", "BRITS", "SAITS", "CSDI"]

# Paths
dataset_fold_path = "C:\\Users\\ryanh\\PycharmProjects\\dim_sum\\benchmark_code\\data\\generated_datasets"
model_results_path = "C:\\Users\\ryanh\\PycharmProjects\\dim_sum\\benchmark_code\\model_results"

# Initialize a dictionary to store results
results = {dataset: {} for dataset in DATASET_PATHS}

# Perform inference using each model on each dataset
for model_name in MODELS:
    for train_dataset_name in DATASET_PATHS:
        # Path to the saved model results for round_4
        model_saving_path = os.path.join(
            model_results_path,
            train_dataset_name,
            f"{model_name}_{train_dataset_name}",
            "round_4"
        )
        imputation_file = os.path.join(model_saving_path, "imputation.pkl")

        if os.path.exists(imputation_file):
            # Load the imputed results
            with open(imputation_file, 'rb') as f:
                imputed_data = pickle.load(f)

            # Assuming the imputed data contains results for the test set
            test_set_imputation = imputed_data["test_set_imputation"]

            # Calculate and store metrics for each dataset
            for test_dataset_name in DATASET_PATHS:
                dataset_path = os.path.join(dataset_fold_path, test_dataset_name)
                # Load all data splits if necessary
                _, _, test_X, test_X_ori, test_indicating_mask = get_datasets_path(dataset_path)

                # Calculate metrics on the test set
                mae = calc_mae(test_set_imputation, test_X_ori, test_indicating_mask)
                mse = calc_mse(test_set_imputation, test_X_ori, test_indicating_mask)
                mre = calc_mre(test_set_imputation, test_X_ori, test_indicating_mask)

                # Store the results
                if test_dataset_name not in results:
                    results[test_dataset_name] = {}
                results[test_dataset_name][f"{model_name}_{train_dataset_name}"] = {
                    "MAE": mae,
                    "MSE": mse,
                    "MRE": mre
                }

                logger.info(
                    f"{model_name} trained on {train_dataset_name} tested on {test_dataset_name}: MAE={mae:.4f}, MSE={mse:.4f}, MRE={mre:.4f}"
                )
        else:
            logger.warning(f"Imputation file not found for {model_name} trained on {train_dataset_name} in round 4.")

# Print the results for comparison
for test_dataset_name, train_results in results.items():
    print(f"Results for dataset {test_dataset_name}:")
    last_model = None
    for train_model_dataset_name, metrics in train_results.items():
        current_model = train_model_dataset_name.split('_')[0]
        if last_model and current_model != last_model:
            print()  # Add a blank line between different models
        print(
            f"  {train_model_dataset_name}: MAE={metrics['MAE']:.4f}, MSE={metrics['MSE']:.4f}, MRE={metrics['MRE']:.4f}"
        )
        last_model = current_model
    print("\n")
import numpy as np
import pickle
import logging
from benchpots.datasets import preprocess_ett
from pygrinder import mcar, seq_missing, block_missing
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_missingness(data, rate, pattern, **kwargs):
    if pattern == 'point':
        return mcar(data, rate)
    elif pattern == 'subseq':
        seq_len = kwargs.get('seq_len', 10)
        return seq_missing(data, rate, seq_len)
    elif pattern == 'block':
        block_size = kwargs.get('block_size', 5)
        return block_missing(data, rate, block_size)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def load_imputation_results(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data['test_set_imputation']


def run_evaluation(imputed_data, original_data, indicating_mask):
    mae = calc_mae(imputed_data, original_data, indicating_mask)
    mse = calc_mse(imputed_data, original_data, indicating_mask)
    mre = calc_mre(imputed_data, original_data, indicating_mask)
    return mae, mse, mre


if __name__ == "__main__":
    # Paths to the imputation results pickle files
    point_imputation_path = "C:\\Users\\ryanh\\PycharmProjects\\dim_sum\\benchmark_code\\rate07_results\\CSDI_ETT_h1\\round_4\\imputation.pkl"
    subseq_imputation_path = "C:\\Users\\ryanh\\PycharmProjects\\dim_sum\\benchmark_code\\rate03_12_ss_results\\CSDI_ETT_h1\\round_4\\imputation.pkl"

    # Initialize lists to store metrics for each iteration
    mae_point_point, mse_point_point, mre_point_point = [], [], []
    mae_subseq_point, mse_subseq_point, mre_subseq_point = [], [], []
    mae_point_subseq, mse_point_subseq, mre_point_subseq = [], [], []
    mae_subseq_subseq, mse_subseq_subseq, mre_subseq_subseq = [], [], []

    for i in range(1000):
        # Generate a new dataset with a 30% missing rate for each iteration
        dataset = preprocess_ett(subset="ETTh1", rate=0.3, n_steps=48, pattern='point')
        test_data_ori = dataset['test_X_ori']

        # Create point-pattern test set
        test_data_point = create_missingness(test_data_ori, rate=0.3, pattern='point')
        test_indicating_mask_point = np.isnan(test_data_point)
        logger.info(f"Iteration {i + 1} - Point Pattern Mask:\n{test_indicating_mask_point.astype(int)}")

        # Create subsequence-pattern test set
        test_data_subseq = create_missingness(test_data_ori, rate=0.3, pattern='subseq', seq_len=24)
        test_indicating_mask_subseq = np.isnan(test_data_subseq)
        logger.info(f"Iteration {i + 1} - Subsequence Pattern Mask:\n{test_indicating_mask_subseq.astype(int)}")

        # Load imputed data
        point_imputed_data = load_imputation_results(point_imputation_path)
        subseq_imputed_data = load_imputation_results(subseq_imputation_path)

        # Evaluate point model on point-pattern test set
        mae, mse, mre = run_evaluation(point_imputed_data, test_data_ori, test_indicating_mask_point)
        mae_point_point.append(mae)
        mse_point_point.append(mse)
        mre_point_point.append(mre)
        logger.info(
            f"Iteration {i + 1} - Point Model on Point-Pattern Test Set: MAE={mae:.4f}, MSE={mse:.4f}, MRE={mre:.4f}")

        # Evaluate subsequence model on point-pattern test set
        mae, mse, mre = run_evaluation(subseq_imputed_data, test_data_ori, test_indicating_mask_point)
        mae_subseq_point.append(mae)
        mse_subseq_point.append(mse)
        mre_subseq_point.append(mre)
        logger.info(
            f"Iteration {i + 1} - Subsequence Model on Point-Pattern Test Set: MAE={mae:.4f}, MSE={mse:.4f}, MRE={mre:.4f}")

        # Evaluate point model on subsequence-pattern test set
        mae, mse, mre = run_evaluation(point_imputed_data, test_data_ori, test_indicating_mask_subseq)
        mae_point_subseq.append(mae)
        mse_point_subseq.append(mse)
        mre_point_subseq.append(mre)
        logger.info(
            f"Iteration {i + 1} - Point Model on Subsequence-Pattern Test Set: MAE={mae:.4f}, MSE={mse:.4f}, MRE={mre:.4f}")

        # Evaluate subsequence model on subsequence-pattern test set
        mae, mse, mre = run_evaluation(subseq_imputed_data, test_data_ori, test_indicating_mask_subseq)
        mae_subseq_subseq.append(mae)
        mse_subseq_subseq.append(mse)
        mre_subseq_subseq.append(mre)
        logger.info(
            f"Iteration {i + 1} - Subsequence Model on Subsequence-Pattern Test Set: MAE={mae:.4f}, MSE={mse:.4f}, MRE={mre:.4f}")

    # Calculate and print the average metrics over all iterations
    logger.info(
        "Point Model on Point-Pattern Test Set - Average MAE: {:.4f}, Average MSE: {:.4f}, Average MRE: {:.4f}".format(
            np.mean(mae_point_point), np.mean(mse_point_point), np.mean(mre_point_point)))
    logger.info(
        "Subsequence Model on Point-Pattern Test Set - Average MAE: {:.4f}, Average MSE: {:.4f}, Average MRE: {:.4f}".format(
            np.mean(mae_subseq_point), np.mean(mse_subseq_point), np.mean(mre_subseq_point)))
    logger.info(
        "Point Model on Subsequence-Pattern Test Set - Average MAE: {:.4f}, Average MSE: {:.4f}, Average MRE: {:.4f}".format(
            np.mean(mae_point_subseq), np.mean(mse_point_subseq), np.mean(mre_point_subseq)))
    logger.info(
        "Subsequence Model on Subsequence-Pattern Test Set - Average MAE: {:.4f}, Average MSE: {:.4f}, Average MRE: {:.4f}".format(
            np.mean(mae_subseq_subseq), np.mean(mse_subseq_subseq), np.mean(mre_subseq_subseq)))
import h5py
import os
import numpy as np

# Define paths to your folders
folder1 = r"C:\Users\ryanh\PycharmProjects\dim_sum\benchmark_code\data\generated_datasets\ett_rate03_step48_subseq_seqlen24"
folder2 = r"C:\Users\ryanh\PycharmProjects\dim_sum\benchmark_code\data\generated_datasets\ett_rate03_step48_point"

# Define output path for combined datasets
output_folder = r"C:\Users\ryanh\PycharmProjects\dim_sum\benchmark_code\data\combined_datasets"
os.makedirs(output_folder, exist_ok=True)


# Function to split and combine datasets
def split_and_combine_datasets(folder1, folder2, output_folder):
    # Create output files
    combined_files = {
        'train': h5py.File(os.path.join(output_folder, 'train.h5'), 'w'),
        'val': h5py.File(os.path.join(output_folder, 'val.h5'), 'w'),
        'test': h5py.File(os.path.join(output_folder, 'test.h5'), 'w')
    }

    # Helper function to split data
    def split_data(data):
        mid_point = len(data) // 2
        return data[:mid_point], data[mid_point:]

    # Iterate over both folders
    for folder in [folder1, folder2]:
        for file_name in os.listdir(folder):
            if file_name.endswith('train.h5'):  # Only process training files
                with h5py.File(os.path.join(folder, file_name), 'r') as h5_file:
                    print(f"File: {file_name}, Datasets: {list(h5_file.keys())}")
                    # Iterate over each dataset key
                    for key in ['X', 'X_ori', 'y']:
                        if key in h5_file:
                            dataset = h5_file[key]
                            if dataset.shape == ():  # Check if the dataset is scalar
                                data1, data2 = dataset[()], dataset[()]  # Duplicate the scalar value
                            else:
                                data = dataset[:]
                                data1, data2 = split_data(data)

                            # Append halves to train, val, and test
                            for combined_key, data_part in zip(['train', 'val', 'test'], [data1, data2, data2]):
                                if key in combined_files[combined_key]:
                                    # Ensure the dataset is resizable
                                    combined_files[combined_key][key].resize(
                                        (combined_files[combined_key][key].shape[0] + (
                                            1 if dataset.shape == () else data_part.shape[0])), axis=0)
                                    combined_files[combined_key][key][-data_part.shape[0]:] = data_part
                                else:
                                    if dataset.shape == ():
                                        # Create a scalar dataset
                                        combined_files[combined_key].create_dataset(key, data=data_part)
                                    else:
                                        # Create a resizable dataset
                                        combined_files[combined_key].create_dataset(
                                            key, data=data_part, maxshape=(None, *data_part.shape[1:]), chunks=True)

    # Close all combined files
    for f in combined_files.values():
        f.close()


split_and_combine_datasets(folder1, folder2, output_folder)

print("Datasets split and combined into train, val, and test successfully.")
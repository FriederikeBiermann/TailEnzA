import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Function to process individual CSV file
def process_csv(file_info):
    file_path, subdir, subsubdir = file_info
    print(file_path)
    df = pd.read_csv(file_path, index_col=0).transpose()
    description = df['description'].iloc[0]
    df['genus'] = subdir
    df['master_record'] = subsubdir.replace("_genomic", "")
    df['description'] = description
    df = df[['ID', 'BGC_type', 'score', 'window_start', 'window_end', 'filename', 'genus', 'master_record', 'description']]
    return df

def collect_file_info(base_dir):
    file_info_list = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv') and (file_path := os.path.join(root, file)) and os.path.getsize(file_path) >= 5 and "result" in file:
                subdir = os.path.basename(os.path.dirname(root))
                subsubdir = os.path.basename(root)
                file_info_list.append((file_path, subdir, subsubdir))
    return file_info_list

# Base directory
base_dir = 'interference_dataset/Output'

# Collect file info
file_info_list = collect_file_info(base_dir)
print(file_info_list)
# Initialize an empty DataFrame
all_data = pd.DataFrame()

# Process files in parallel
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_csv, file_info_list))

# Concatenate all the DataFrames
all_data = pd.concat(results, ignore_index=True)

# Save the concatenated DataFrame to the highest-order directory
output_path = os.path.join(base_dir, 'concatenated_results_df_final.csv')
all_data.to_csv(output_path, index=False)


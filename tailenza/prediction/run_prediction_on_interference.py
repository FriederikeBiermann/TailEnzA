import os
import subprocess
import concurrent.futures

# Define the root directory for input and output
input_root = 'interference_dataset/output'
output_root = 'interference_dataset/Output'
failed_files_log = 'failed_files.txt'

# Create the output root directory if it doesn't exist
if not os.path.exists(output_root):
    os.makedirs(output_root)

# Function to process a single file
def process_file(input_file, output_file):
    output_file = f"{os.path.splitext(output_file)[0]}/"
    command = [
        'python3', 'prediction_module.py',
        '-i', input_file,
        '-c', '0.5',
        '-o', output_file
    ]
    print(command)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        return input_file
    return None

# Collect all files to process
file_pairs = []
for subdir, _, files in os.walk(input_root):
    # Create corresponding output directory
    relative_path = os.path.relpath(subdir, input_root)
    output_subdir = os.path.join(output_root, relative_path)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    for file in files:
        input_file = os.path.join(subdir, file)
        output_file = os.path.join(output_subdir, file)
        file_pairs.append((input_file, output_file))

# Use ThreadPoolExecutor to process files with multiple workers
failed_files = []
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_file, input_file, output_file) for input_file, output_file in file_pairs]

    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result:
            failed_files.append(result)

# Write out the failed files
if failed_files:
    with open(failed_files_log, 'w') as f:
        for failed_file in failed_files:
            f.write(f"{failed_file}\n")

print(f"Processing complete. Failed files are logged in {failed_files_log}.")


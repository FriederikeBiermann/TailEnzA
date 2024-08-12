import os
import shutil
from pathlib import Path
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
def process_file(input_file, output_file, device):
    output_file = f"{os.path.splitext(output_file)[0]}/"
    log_file = f"{os.path.splitext(output_file)[0]}log.log"
    command = [
        'python3', 'prediction_module.py',
        '-i', input_file,
        '-c', '0.0',
        '-o', output_file,
        '-d', f"cuda:{device}"
    ]
    print(command)
    Path(os.path.splitext(output_file)[0]).mkdir(parents=True, exist_ok=True)

    def run_command():
        try:
            with open(log_file, 'w') as lf:
                result = subprocess.run(command, check=True, stdout=lf, stderr=lf)
            return True
        except subprocess.CalledProcessError as e:
            return False, str(e)

    success, error_message = run_command(), ""
    if not suqccess:
        # Try running the command once more if it fails
        print("Retrying...")
        success, error_message = run_command()

    if not success:
        with open(log_file, 'a') as lf:
            lf.write(f"Error processing {input_file}: {error_message}\n")
        return input_file

    return None

def is_valid_directory(subdir):
    files = os.listdir(subdir)
    has_log = 'log.log' in files
    has_csv = any(file.endswith('.csv') for file in files)
    return has_log and not has_csv

# Collect all directories to process one level deeper
dirs_to_process = []
for subdir, dirs, files in os.walk(output_root):
    for sub_subdir in dirs:
        full_sub_subdir_path = os.path.join(subdir, sub_subdir)
        print(full_sub_subdir_path)
        if is_valid_directory(full_sub_subdir_path):
            dirs_to_process.append(full_sub_subdir_path)
print(dirs_to_process)
# Remove all contents of the directories before processing
for dir_to_process in dirs_to_process:
    for item in os.listdir(dir_to_process):
        item_path = os.path.join(dir_to_process, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

print(dirs_to_process)
# Collect all files to process
file_pairs = []
for subdir in dirs_to_process:
    relative_path = os.path.relpath(subdir, input_root)
    output_subdir = os.path.join(output_root, relative_path)
    
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    input_file = os.path.join(input_root, relative_path)
    output_file = os.path.join(output_root, relative_path)
    file_pairs.append((input_file, output_file))

print(file_pairs)
exit()

device_assignment = [0, 1]  # Assuming you have two devices: cuda:0 and cuda:1
device_index = 0
# Use ThreadPoolExecutor to process files with multiple workers
failed_files = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for input_file, output_file in file_pairs:
        device = device_assignment[device_index % len(device_assignment)]
        futures.append(executor.submit(process_file, input_file, output_file, device))
        device_index += 1

    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result:
            failed_files.append(result)

# Write the failed files to the log
if failed_files:
    with open(failed_files_log, 'w') as f:
        for file in failed_files:
            f.write(f"{file}\n")


import os
from pathlib import Path
import subprocess
import concurrent.futures

# Define the root directory for input and output
input_root = '/lustre/BIF/nobackup/bierm010/TailEnzA/tailenza/prediction/thao/annotated_files'
output_root = '/lustre/BIF/nobackup/bierm010/TailEnzA/tailenza/prediction/thao/tailenza_output'

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
    if not success:
        # Try running the command once more if it fails
        print("Retrying...")
        success, error_message = run_command()
    
    if not success:
        with open(log_file, 'a') as lf:
            lf.write(f"Error processing {input_file}: {error_message}\n")
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

device_assignment = [0, 1]  # Assuming you have two devices: cuda:0 and cuda:1
device_index = 0
# Use ThreadPoolExecutor to process files with multiple workers
failed_files = []
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
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


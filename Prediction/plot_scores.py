import os
import re
import shutil
import matplotlib.pyplot as plt

def extract_score(filename):
    match = re.search(r'^(\d+\.\d+)', filename)
    return float(match.group(1)) if match else None

def create_dir_and_move(file, dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    shutil.move(file, dir_name)

def main():
    scores = []
    for file in os.listdir('.'):
        if file.endswith('.gb'):
            score = extract_score(file)
            if score is not None:
                scores.append(score)
                interval = round(score / 0.1) * 0.1
                dir_name = f"{interval:.1f}"
                create_dir_and_move(file, dir_name)

    # Plotting the histogram
    plt.hist(scores, bins='auto')
    plt.title('Histogram of Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    main()
 

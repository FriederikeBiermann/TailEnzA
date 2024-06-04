import csv
import os

pfam_hmm_database = "/Users/theohaas/Desktop/MBW_Master/2._Semester/Synthese_Naturstoffe/datasets/Pfam-A.hmm"
output_directory = "/Users/theohaas/Desktop/MBW_Master/2._Semester/Synthese_Naturstoffe/output"
tsv_file = "/Users/theohaas/Desktop/MBW_Master/2._Semester/Synthese_Naturstoffe/entries.tsv"

# Dictionary to store group names as keys and corresponding PFAM domains as values
groups = {
    "methyltransferase": [],
    "ycao": ["PF02624", "PF18381"],
    "P450": ["PF00067"],
    "radical SAM": ["PF04055"]
}

# Open the TSV file and read its contents
with open(tsv_file, "r") as file:
    tsv_reader = csv.reader(file, delimiter="\t")

    # Iterate over each row in the TSV file


    for row in tsv_reader:
        # Extract the PFAM domain from the first column
        pfam_domain = row[0]
        if pfam_domain.startswith("PF"):
            # Add the PFAM domain to the list
            groups["methyltransferase"].append(pfam_domain)

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Open the HMM database file and process each line
with open(pfam_hmm_database, "r") as input_file:
    lines = input_file.read().split("//")
    for group_name, domains in groups.items():
        # Create a file path for the group's HMM
        output_file = os.path.join(output_directory, f"{group_name}.hmm")
        with open(output_file, "w") as output:
            for line in lines:
                if any(domain in line for domain in domains):
                    output.write(line + "//")

print("Extraction complete.")

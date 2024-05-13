 
import os
import csv
import json
from Bio import SeqIO

def is_csv_empty(csv_file):
    """
    Check if the CSV file is empty.
    """
    with open(csv_file, 'r') as file:
        return not bool(len(file.readline()))

def find_csv_files_for_record(record_id, base_csv_dir):
    """
    Find all CSV files in the subdirectory corresponding to a given record ID.
    """
    csv_dir = os.path.join(base_csv_dir, record_id)
    if not os.path.exists(csv_dir):
        return []
    return [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]

def aggregate_csv_data(csv_files):
    """
    Aggregate data from multiple CSV files into a dictionary.
    """
    aggregated_data = {}
    for csv_file in csv_files:
        if not is_csv_empty(csv_file):
            with open(csv_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Parse the 'protein_details' field as JSON
                    protein_details = json.loads(row['protein_details'])
                    for detail in protein_details:
                        protein_id = detail['protein_id']
                        aggregated_data[protein_id] = detail
    return aggregated_data

def add_qualifiers_to_record(record, csv_data):
    """
    Update CDS features for a given record with CSV data.
    """
    for feature in record.features:
        if feature.type == "CDS":
            protein_id = feature.qualifiers.get('protein_id', [''])[0]
            if protein_id in csv_data:
                for key, value in csv_data[protein_id].items():
                    feature.qualifiers[key] = [value]
    return record

def process_genbank_file(genbank_file, base_csv_dir):
    # Load GenBank file and iterate over each record
    records = SeqIO.parse(genbank_file, "genbank")
    updated_records = []

    for record in records:
        # Find all corresponding CSV files in the subdirectory
        record_id = os.path.splitext(os.path.basename(genbank_file))[0]
        csv_files = find_csv_files_for_record(record_id, base_csv_dir)

        # Aggregate data from CSV files
        csv_data = aggregate_csv_data(csv_files)

        # Update CDS features in the record
        updated_record = add_qualifiers_to_record(record, csv_data)
        updated_records.append(updated_record)

    # Write updated records to a new GenBank file
    updated_file = os.path.splitext(genbank_file)[0] + "_updated.gbff"
    SeqIO.write(updated_records, updated_file, "genbank")

# Example usage
genbank_dir = 'path_to_your_genbank_files'
base_csv_dir = 'path_to_your_csv_directory'
for filename in os.listdir(genbank_dir):
    if filename.endswith("_genomic.gbff"):
        genbank_file = os.path.join(genbank_dir, filename)
        process_genbank_file(genbank_file, base_csv_dir)

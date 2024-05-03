import setuptools
import os
import requests

def download_file(url, filename):
    """Helper function to download a file from a URL."""
    if not os.path.isfile(filename):
        print(f"{filename} not found, downloading...")
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename}")
        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")  # Handle HTTP errors like 403, 404, etc.
        except requests.exceptions.RequestException as err:
            print(f"Error downloading {filename}: {err}")  # Handle other possible errors

    else:
        print(f"{filename} already exists.")

# Define the filenames and their URLs
files = {
    "tailenza/data/esm1b_t33_650M_UR50S.pt": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt",
    "tailenza/data/esm1b_t33_650M_UR50S-contact-regression.pt": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt"
}

# Check each file and download if not present
for filename, url in files.items():
    download_file(url, filename)

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setup configuration
setuptools.setup(
    name="tailenza",
    version="0.0.1",
    author="FriederikeBiermann",
    author_email="friederike@biermann-erfurt.de",
    description="A package for the prediction of tailoring enzyme affiliation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FriederikeBiermann/TailEnzA",
    project_urls={
        "Bug Tracker": "https://github.com/FriederikeBiermann/TailEnzA/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        "tailenza": ["data/*.pt"],
    },
    python_requires=">=3.6",
        install_requires=[
        "biopython==1.7.8",
        "scikit-learn==1.4.2",
        "imbalanced-learn==0.12.2",
        "pandas==2.2.2",
        "fair-esm==2.0.0",
        "torch==1.8.0",
    ],
    packages=find_packages()
)

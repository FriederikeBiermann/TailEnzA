import setuptools
import os
import requests

def download_file(url, filename):
    """Helper function to download a file from a URL."""
    if not os.path.isfile(filename):
        print(f"{filename} not found, downloading...")
        response = requests.get(url)
        response.raise_for_status()  # Will raise an exception on a bad response
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists.")

# Define the filenames and their URLs
files = {
    "data/esm1b_t33_650M_UR50S-contact-regression.pt": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S-contact-regression.pt",
    "data/esm1b_t33_650M_UR50S.pt": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt"
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
)

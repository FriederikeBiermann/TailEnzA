# TailEnzA Installation Guide

### Background
TailEnzA was developed to enhance genome mining by leveraging ubiquitous tailoring enzymes as markers to identify novel biosynthetic gene clusters. Unlike traditional genome mining algorithms that rely on detecting core biosynthetic genes and Pfam signatures, TailEnzA aims to predict substrate specificity and infer the presence of gene clusters. By training an algorithm to analyze tailoring enzymes, the approach improves the identification of non-canonical BGCs that may lack well-known encoded enzymes.

---


## Installation Steps

### 1. Clone the Repository
```bash
$ git clone https://github.com/FriederikeBiermann/TailEnzA.git
$ cd TailEnzA
```

### 2. Set Up a Conda Environment


```bash
$ conda create --name TailEnzA python=3.9.6
```

Activate the newly created environment:
```bash
$ conda activate TailEnzA
```

### 3. Install TailEnzA and Dependencies
Install the package and dependencies:
```bash
$ apt install muscle3
$ pip install .
```


## Usage
After installation, you can start using TailEnzA by using the prediction_module.py script on your genbank files. Please make sure the genome has genes CDS already annotated, otherwiese use an annotation pipeline like Prodigal.
```bash
python prediction_module.py -i (input_dir) -o (ouput_dir) -c (cutoff, default = 0.3)
```





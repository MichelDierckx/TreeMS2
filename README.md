# TreeMS2

## Description

An efficient tool for phylogenetic analysis for ms/ms spectra using ANN indexing.

## Usage

### Requirements

Install requirements via:

```bash
pip install -r requirements.txt
```

### Configuration

To request an overview of the available parameters:

```bash
python3 -m TreeMS2.tree_ms2 --help
```

### Running the program

Run the program by either providing a config.ini file or by providing the required arguments:

```bash
python3 -m TreeMS2.tree_ms2 -c /path/to/config.ini
```

### Running tests

```bash
python3 -m unittest discover -s tests
```
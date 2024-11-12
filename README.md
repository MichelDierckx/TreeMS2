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
python3 -m TreeMS2 --help
```

### Running the program

Run the program by either providing a config.ini file or by providing the required arguments:

```bash
python3 -m TreeMS2 -c config.ini
```

### Running tests

```bash
python3 -m unittest discover -s tests
```

## Currently implemented steps

1. Data linking ms/ms peak files to certain groups or species is extracted.
2. The spectra in the ms/ms peak files are processed and filtered:
    1. Restrict the m/z range to a minimum and maximum m/z.
    2. Remove peak(s) around the precursor m/z value.
    3. Remove peaks below a percentage of the base peak intensity.
    4. Retain only the top most intense peaks.
    5. Scale and normalize peak intensities.
    6. Discard spectrum if low quality (not enough peaks covering a wide enough mass
       range)
    7. Peak data is converted to lower dimensionality binned sparse vectors.
    8. Write spectrum data together with group and file information to Lance dataset.
        
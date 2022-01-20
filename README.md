# General Program Synthesis Benchmark Suite Datasets

Version 1.0.1 (see version history at bottom)

This repository contains datasets for the 25 problems described in the paper *PSB2: The Second Program Synthesis Benchmark Suite*. These problems come from a variety of sources, and require a range of programming constructs and datatypes to solve. These datasets are designed to be usable for any method of performing general program synthesis, including and not limited to inductive program synthesis and evolutionary methods such as genetic programming.

For more information, see the associated website: https://cs.hamilton.edu/~thelmuth/PSB2/PSB2.html

## Use

Each problem in the benchmark suite is located in a separate directory in the `datasets` directory.

For each problem, we provide a set of `edge` cases and a set of `random` cases. The `edge` cases are hand-chosen cases representing the limits of the problem. The `random` cases are all generated based on problem-specific distributions. For each problem, we included exactly 1 million `random` cases.

A typical use of these datasets for a set of runs of program synthesis would be:

- For each run, use every `edge` case in the training set
- For each run, use a different, randomly-sampled set of `random` cases in the training set.
- Use a larger set of `random` cases as an unseen test set.

## Sampling Libraries

We provide the following libraries to make the downloading and sampling of these datasets easier. Using these libraries, you do not need to download the entire dataset from Zenodo; the individual problem datasets are downloaded and stored once when first sampling them.

- Python: https://github.com/thelmuth/psb2-python
- Clojure: https://github.com/thelmuth/psb2-clojure

## Dataset format

Each edge and random dataset is provided in three formats: CSV, JSON, and EDN, with all three formats containing identical data.

The CSV files are formatted as follows:

- The first row of the file is the column names.
- Each following row corresponds to one set of program inputs and expected outputs.
- Input columns are labeled `input1`, `input2`, etc., and output columns are labeled `output1`, `output2`, etc.
- In CSVs, string inputs and outputs are double quoted when necessary, but not if not necessary. Newlines within strings are escaped.
- Columns in CSV files are comma-separated.

The JSON and EDN files are formatted using the [JSON Lines](https://jsonlines.org/) standard (adapted for EDN).
Each case is put on its own line of the data file. The files should be read line-by-line and each parsed into an object/map using a JSON/EDN parser.

## Citation

If you use these datasets in a publication, please cite the paper *PSB2: The Second Program Synthesis Benchmark Suite* and include a link to this repository.

BibTeX entry for paper:

```bibtex
@InProceedings{Helmuth:2021:GECCO,
  author =	"Thomas Helmuth and Peter Kelly",
  title =	"{PSB2}: The Second Program Synthesis Benchmark Suite",
  booktitle =	"2021 Genetic and Evolutionary Computation Conference",
  series = {GECCO '21},
  year = 	"2021",
  isbn13 = {978-1-4503-8350-9},
  address = {Lille, France},
  size = {10 pages},
  doi = {10.1145/3449639.3459285},
  publisher = {ACM},
  publisher_address = {New York, NY, USA},
  month = {10-14} # jul,
  doi-url = {https://doi.org/10.1145/3449639.3459285},
  URL = {https://dl.acm.org/doi/10.1145/3449639.3459285},
}
```

## Version History

1.0.0 - 2021/4/10 - Initial publication of PSB2 datasets on Zenodo.

1.0.1 - 2021/7/9 - Changes to CSVs to quote all strings that could be read as integers. No changes in actual data, just formatting.

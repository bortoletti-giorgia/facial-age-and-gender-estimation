# Guess the age

## Repository Structure
```
.
├── dataset                       # Dataset files
│   ├── analyse-adience.ipynb     # Do analysis on age and gender distribution in Adience dataset
│   ├── analyse-imdbwiki.ipynb    # Do analysis on age and gender distribution in IMDB-WIKI dataset
│   ├── analyse-utkface.ipynb     # Do analysis on age and gender distribution in UTKFace dataset
│   ├── create-csv-adience.py     # Create CSV file from Adience dataset in DeepLake
│   ├── create-csv-imdbwiki.py    # Create CSV file from local IMDB-WIKI dataset
│   ├── create-csv-utkface.py     # Create CSV file from local UTKFace dataset
│   ├── add-age-group.py          # Add 'age-group' column in each CSV file
│   ├── prepare-images.ipynb      # Prepare datasets for StyleGan2-ADA-Pytorch
│   └── unit                # Unit tests
├── docs                    # Documentation files
├── test                    # Automated tests (alternatively `spec` or `tests`)
├── tools                   # Tools and utilities
├── LICENSE
└── README.md
```
## Dataset

Chosen training datasets are: UTKFace, IMDB-WIKI and Adience.

Some demographic analyses are under the folder "dataset".
Use [prepare-images.ipynb](https://github.com/bortoletti-giorgia/facial-age-estimation/blob/dataset/dataset/prepare-images.ipynb) to prepare dataset for StyleGan2-ADA-Pytorch.

## Model


## Evaluation
 

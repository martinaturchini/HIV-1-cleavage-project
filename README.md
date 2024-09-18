# HIV-1-cleavage-prediction
This project applies supervised machine learning to predict HIV-1 cleavage sites. It compares the performance of three classifiers: Multi-Layer Perceptron (MLP), Logistic Regression, and k-Nearest Neighbors (k-NN).

# Description
The goal of this project is to study and predict HIV-1 cleavage sites using sequence information, including amino acid binary profiles, bond composition, and physicochemical properties. The Multi-Layer Perceptron (MLP) classifier achieves the best performance, while Logistic Regression produces similar results. The k-NN classifier shows the lowest accuracy among the models analyzed. Future improvements could be made by utilizing a larger dataset or applying data augmentation techniques.

# Usage
To use this project in a Jupyter Notebook, download and save the dataset from [this link](https://github.com/martinaturchini/HIV-1-cleavege-/blob/main/12859_2022_5017_MOESM2_ESM.xlsx) into the following location:
```bash
/content/gdrive/My Drive
```

# Runnig the tests
To test the functionality of the project, follow these steps:
1. Clone the project repository:
```bash
git clone https://github.com/martinaturchini/HIV-1-cleavage-project.git
```
2. Navigate into the project directory:
```bash
cd HIV-1-cleavage-project/
```
3. Run all the tests:
```bash
python3 -m unittest discover -s tests
```

# Reference
E. Onah, P. F. Uzor, I. C. Ugwoke, et al., “Prediction of HIV-1 protease cleavage site from octapeptide sequence information using selected classifiers and hybrid descriptors”, BMC Bioinformatics 23, 10.1186/s12859-022-05017-x (2022).

# HIV-1-cleavage-prediction
Supervised Machine Learning project for HIV-1 cleavege sites. This project compares the results obtained with three different classifiers: Multi-Layer Perceptron, Logistic Regression and k-NN.

# Description
This work shows the possibility to study and predict the HIV-1 cleavage sites given the combination of sequence information, including amino acid binary profiles, bond composition, and physicochemical properties. The best performances are obtained via the Multi-Layer Perceptron classifier and similar results can be obtained with the Logistic Regression classifier. The k-NN classifier is the worst among the methods analyzed in this work. The results could be improved by using a larger data set, that can be implemented via data augmentation techniques.

# Usage
To use this project in a Jupyter Notebook, you need to download and save the dataset from [this link](https://github.com/martinaturchini/HIV-1-cleavege-/blob/main/12859_2022_5017_MOESM2_ESM.xlsx) into the following location:
```bash
/content/gdrive/My Drive
```

# Run the tests
You can clone the project repository using the following command:
```bash
git clone https://github.com/martinaturchini/HIV-1-cleavage-project.git
```
After cloning, navigate into the project directory:
```bash
cd HIV-1-cleavage-project/
```
To run all the tests, use this command:
```bash
python3 -m unittest discover -s tests
```

# Reference
E. Onah, P. F. Uzor, I. C. Ugwoke, et al., “Prediction of HIV-1 protease cleavage site from octapeptide sequence information using selected classifiers and hybrid descriptors”, BMC Bioinformatics 23, 10.1186/s12859-022-05017-x (2022).

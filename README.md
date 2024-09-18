# HIV-1-cleavage-prediction
Supervised Machine Learning project for HIV-1 cleavege sites. This project compares the results obtained with three different classifiers: Multi-Layer Perceptron, Logistic Regression and k-NN.

# Description
This work shows the possibility to study and predict the HIV-1 cleavage sites given the combination of sequence information, including amino acid binary profiles, bond composition, and physicochemical properties. The best performances are obtained via the Multi-Layer Perceptron classifier and similar results can be obtained with the Logistic Regression classifier. The k-NN classifier is the worst among the methods analyzed in this work. The results could be improved by using a larger data set, that can be implemented via data augmentation techniques.

# Usage
To work with the code on Jupyter Notebook is necessary to download and save the data set (https://github.com/martinaturchini/HIV-1-cleavege-/blob/main/12859_2022_5017_MOESM2_ESM.xlsx) in "/content/gdrive/My Drive".
For the testing you can download the project directory with: git clone https://github.com/martinaturchini/HIV-1-cleavage-project.git
Then, moving in the "HIV-1-cleavage-project/" directory,you can test all the functions at once with: python3 -m unittest discover -s tests


# Reference
E. Onah, P. F. Uzor, I. C. Ugwoke, et al., “Prediction of HIV-1 protease cleavage site from octapeptide sequence information using selected classifiers and hybrid descriptors”, BMC Bioinformatics 23, 10.1186/s12859-022-05017-x (2022).

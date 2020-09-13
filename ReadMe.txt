#  Genre Prediction using Gutenberg Corpus

This file is a description for all the efforts for the ATML project on the Gutenberg corpus completed by Group 1. We have included the project report and the source code. Project has been implemented using Python3 and all the libraries used are mentioned in the project report. The code has comments all along, and the references have also been duly mentioned everywhere. 

This folder is organised as follows:

-Project Report: Group1_GenreClassification_ProjectReport.pdf

________________

-Folders:

Step 1. Feature Engineering: contains all code for preprocessing and cleaning the data. Prepping it for the machine learning pipeline. It also contains the code for the hand-crafted features that we extracted from the given corpus. Some features required extensive processing, and all code with the necessary comments are in this folder. The files are named in a self-explanatory fashion ex. The TTR code with chunks, gender detection with Glove, and so on. We have also included our preliminary EDA code in this folder itself. 

Step 2. Models: This folder contains the source code for the various models we tested for our data namely Naive Bayes, SVM, and Logistic Regression. Each model was tested with the Doc to Vec representation, TF-IDF and against all features as well. The TF-IDF model for SVM and GNB is is one notebook present in the SVM folder file named 'Tf_idf_SVM-MNB.ipynb'.

Step 3. Reports: This folder contains the reports and the visualisation we created for this project. All the model classification reports, comparison plots, excel documents comparing the model across all three approaches and their precision, recall, f1-scores etc. 

- Main focus : feature engineering ie. finding the best representation for a book for a classification task, etc. The use of precision as a metric of evaluation since accuracy is not a reliable source, the recall could have been 100 percent but the precision is a better measure for the problem at hand. Use of Glove libraries for gender prediction because of its extensively trained library and excellent results. 

________________

- Members

  Manali Thakur
  Saloni Verma
  Shubham Singh
  Shweta Bhat
  Surabhi Katti


We followed all steps of the scientific project and have aimed to deliver all the requirements of the team task. 
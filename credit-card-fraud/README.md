# Credit Card Fraud Detection using LSTM Network

<p align="center">
  <img src="../metadata/gif/creditcard.gif">
</p>

## Overview 

This dataset is highly unbalanced, adding complexity to the data modelling.


## File Structure 
```
credit-card-fraud 
│   
│───src/main/java/com/codenamewei/CreditCardFraud/App.java    
│
│───PrePostProcessing   
│└─────DataPreprocessing
│     │   Credit-Card-Data-Cleaning_0.ipynb
│     │   Credit-Card-Data_Splitting_1.ipynb
│   
└ ─────ResultPostProcessing
      │   Credit-Card-Result-Visualization.ipynb
```

**App.java:**
 
**Credit-Card-Data-Cleaning_0.ipynb:**  
Remove first row of the data(column names) and first column of the data(Time feature).
 
**Credit-Card-Data_Splitting_1.ipynb:**  
- Normalize the total transaction column of the data
- Split data points into feature and labels and store as .csv.

**Credit-Card-Result-Visualization.ipynb:**
Visualize the results of the testing data and plot recall-precision curve.  

## Data Sources
Data is retrieved from https://www.kaggle.com/mlg-ulb/creditcardfraud.  
Preprocessing steps below are done to obtain the current CreditCardFraud.zip data for training and testing.    

(1) Drop the time column (For modelling with LSTM, time feature is not necessary)  
(2) Drop the row with name of the columns (Time, V1, V2, V3 ...) 
(3) Normalize the total amount transaction column into the range [-1, 1].   
(4) Split each data point into feature(.csv) and label(.csv) and place in separate directories according to the label  
(5) Redistribute the files into training and testing data sets.  

Each data point(.csv) have a file(.csv) with the same name in the *_label.  
For example, train_data_feature/0.csv<-->train_data_label/0.csv

### Data Structure of CreditCardFraud.zip

Data structure and distribution in CreditCardFraud.zip is elaborated as follows
```
CreditCardFraud.zip
│
└───train_data_feature
│   │   0.csv
│   │   1.csv
│   │   ...
│   │   99999.csv
│   
└───train_data_label
│   │   0.csv
│   │   1.csv
│   │   ...
│   │   99999.csv
│   
└───test_data_feature
│   │   0.csv
│   │   1.csv
│   │   ...
│   │   8489.csv
│   
└───test_data_label
│   │   0.csv
│   │   1.csv
│   │   ...
│   │   8489.csv
```
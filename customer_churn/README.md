# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project build an ML model to identify customers that are most likely to churn. It consists of
the python module to load, process and train ML model to identify customers. Further, the project
also has unit tests with logging to check if the python module is working properly.

## Files and data description
<pre>
Folder Structure
.
├── churn_library.py - Main python module to run train the model
├── churn_script_logging_and_tests.py - Unit test and logging script
├── data
│   └── bank_data.csv
├── Guide.ipynb
├── images
│   ├── eda
│   │   ├── Churn.jpg
│   │   ├── Customer_Age.jpg
│   │   ├── Heatmap.jpg
│   │   ├── Marital_Status.jpg
│   │   └── Total_Trans.jpg
│   └── results
│       ├── logistic_regression_classification_report.jpg
│       ├── random_forest_classification_report.jpg
│       └── random_forest_feat_imp.jpg
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── README.md
└── requirements_py3.8.txt
</pre>


## Running Files
<pre>
Step 1: Create Virtual Environment:
    - conda create --name churn_predict python=3.6
    - conda activate churn_predict
Step 2 : Install packages
    - conda install --file requirements.txt
Step 3 : Train Churn Model
    - python churn_library.py
Step 4 : Test churn prediction[optional]
    - python churn_script_logging_and_tests.py
</pre>

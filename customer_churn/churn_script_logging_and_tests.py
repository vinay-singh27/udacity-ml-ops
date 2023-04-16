"""
Module encapsulating the unit tests to
test the functions present in the churn library
Author: Vinay Singh
Creation Date: 08/02/2022
"""
# import libraries
import os
import logging
import churn_library as cls

# load required inputs
CAT_COLUMNS = ['Gender', 'Education_Level',
               'Marital_Status', 'Income_Category', 'Card_Category']
INPUT_FEATURES = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn',
    'Income_Category_Churn',
    'Card_Category_Churn']
PARAM_GRID = {'n_estimators': [200, 500],
              'max_features': ['auto', 'sqrt'],
              'max_depth': [4, 5, 100],
              'criterion': ['gini', 'entropy']
              }
input_df = cls.import_data("data/bank_data.csv")
X_train, X_test, label_train, label_test = cls.perform_feature_engineering(
    input_df, categorical_features=CAT_COLUMNS, input_features=INPUT_FEATURES)

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    """
    test perform eda - function to test whether to perform EDA is
    giving the expected output
    """
    try:
        perform_eda(input_df)
        assert os.path.isfile('images/eda/Churn.jpg')
        assert os.path.isfile('images/eda/Customer_Age.jpg')
        assert os.path.isfile('images/eda/Marital_Status.jpg')
        assert os.path.isfile('images/eda/Total_Trans.jpg')
        assert os.path.isfile('images/eda/Heatmap.jpg')
        logging.info(
            "All the files are generated correctly by perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "The required files were not generated correctly by perform_eda")
        raise err


def test_encoder_helper(encoder_helper):
    """
    test encoder helper - It checks if for all the categorical columns
    the respective churn columns are created or NOT
    """
    try:
        result_df = encoder_helper(input_df, CAT_COLUMNS)
        churn_cols = [col for col in result_df.columns if "_Churn" in col]
        assert len(churn_cols) == len(CAT_COLUMNS)
        logging.info(
            "All the categorical columns respective churn columns are generated: SUCCESS")
    except AssertionError as err:
        logging.error(
            "The categorical columns' churn columns are NOT generated: ERROR")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform feature engineering function
    """

    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            input_df, categorical_features=CAT_COLUMNS, input_features=INPUT_FEATURES)
        assert x_train.shape[0] < input_df.shape[0]
        assert y_train.shape[0] < input_df.shape[0]
        assert x_test.shape[0] < x_train.shape[0]
        assert y_test.shape[0] < x_train.shape[0]
        logging.info(
            "The function has created train & test datasets successfully: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Error while running the perform_feature_engineering function")
        raise err


def test_train_models(train_models):
    """
    test train models and validate if all the input files are generated
    """
    try:
        train_models(X_train, X_test, label_train, label_test, PARAM_GRID)
        assert os.path.isfile('models/rfc_model.pkl')
        assert os.path.isfile('models/logistic_model.pkl')
        assert os.path.isfile(
            'images/results/logistic_regression_classification_report.jpg')
        assert os.path.isfile(
            'images/results/random_forest_classification_report.jpg')
        assert os.path.isfile('images/results/random_forest_feat_imp.jpg')
        logging.info(
            "The function has run generated all the required files : SUCCESS ")
    except Exception as err:
        logging.error(
            "Error while running the perform_feature_engineering function")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

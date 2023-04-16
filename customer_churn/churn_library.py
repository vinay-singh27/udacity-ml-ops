# library doc string
"""
Churn Library which encapsulates the required
functions to perform feature engineering and train
the ML model
Author: Vinay Singh
Creation Date: 08/02/2022
"""
# import libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_curve

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()
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


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data: pandas dataframe
    """
    data = pd.read_csv(pth)
    # create churn flag
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data


def perform_eda(dataframe):
    """
    perform eda on df and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    """
    col_list = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans",
        "Heatmap"]
    for col in col_list:
        plt.figure(figsize=(20, 10))
        if col == "Churn":
            dataframe.Churn.hist()
        elif col == "Customer_Age":
            dataframe.Customer_Age.hist()
        elif col == "Marital_Status":
            dataframe.Marital_Status.value_counts("normalize").plot(kind="bar")
        elif col == "Total_Trans":
            sns.distplot(dataframe.Total_Trans_Ct)
        elif col == "Heatmap":
            sns.heatmap(
                dataframe.corr(),
                annot=False,
                cmap="Dark2_r",
                linewidths=2)
        plt.savefig(f"images/eda/{col}.jpg")
        plt.close()


def encoder_helper(dataframe, category_lst):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]
    output:
            df: pandas dataframe with new columns for
    """
    # encode all category cols
    for col in category_lst:
        col_lst = []
        col_groups = dataframe.groupby(col).mean()['Churn']
        for val in dataframe[col]:
            col_lst.append(col_groups.loc[val])
        dataframe[f'{col}_Churn'] = col_lst
    return dataframe


def perform_feature_engineering(
        dataframe,
        categorical_features,
        input_features):
    """
    input:
          dataframe: pandas dataframe
          response: string of response name [optional argument that
          could be used for naming variables or index y column]

    output:
          X_train: X training data
          X_test: X testing data
          y_train: y training data
          y_test: y testing data
    """
    # perform one-hot encoding of categorical features
    processed_df = encoder_helper(dataframe, categorical_features)
    # create X & y
    x_data = pd.DataFrame()
    x_data[input_features] = processed_df[input_features]
    y_data = processed_df['Churn']
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(result_dict, model_name):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:  result_dict with the following keys:
                y_train: training response values
                y_test:  test response values
                y_train_preds: training predictions from model
                y_train_preds: training predictions from model

    output:
             None
    """
    # extract info
    y_train = result_dict["y_train"]
    y_test = result_dict["y_test"]
    y_train_pred = result_dict["y_train_pred"]
    y_test_pred = result_dict["y_test_pred"]
    # plot classification report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str(f'{model_name} Train'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_pred)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str(f'{model_name} Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_pred)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(f"images/results/{model_name}_classification_report.jpg")
    plt.close()


def plot_roc_curve(y_true, y_pred_rf, y_pred_lg):
    """
    plots the roc curve based of the probabilities
    """
    plt.figure(figsize=(20, 5))
    fpr, tpr, _ = roc_curve(y_true, y_pred_rf)
    plt.plot(fpr, tpr)
    fpr, tpr, _ = roc_curve(y_true, y_pred_lg)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("images/results/roc_curves.jpg")
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importance in pth
    input:
            model: model object containing feature_importance_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importance
    importance = model.feature_importances_
    # Sort feature importance in descending order
    indices = np.argsort(importance)[::-1]
    # Rearrange feature names so they match the sorted feature importance
    names = [x_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(x_data.shape[1]), importance[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()


def train_models(x_train, x_test, y_train, y_test, param_grid):
    """
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # random forest
    rfc = RandomForestClassifier(random_state=42)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    rfc_model = cv_rfc.best_estimator_
    y_train_preds_rf = rfc_model.predict(x_train)
    y_test_preds_rf = rfc_model.predict(x_test)
    rfc_result_dict = {
        "model": rfc_model,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_preds_rf,
        "y_test_pred": y_test_preds_rf}
    classification_report_image(rfc_result_dict, "random_forest")

    # logistic regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(x_train, y_train)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    lg_result_dict = {
        "model": lrc,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_preds_lr,
        "y_test_pred": y_test_preds_lr}
    classification_report_image(lg_result_dict, "logistic_regression")
    feature_importance_plot(cv_rfc.best_estimator_, x_train,
                            './images/results/random_forest_feat_imp.jpg')

    # plot roc curves
    plot_roc_curve(y_test, y_test_preds_rf, y_test_preds_lr)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    input_data = import_data("data/bank_data.csv")
    perform_eda(input_data)
    X_train, X_test, label_train, label_test = perform_feature_engineering(
        input_data, categorical_features=CAT_COLUMNS, input_features=INPUT_FEATURES)
    train_models(X_train, X_test, label_train, label_test, PARAM_GRID)

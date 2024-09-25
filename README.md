# Bank Product Mapping & Overfitted Model Prediction

This project uses an overfitted `RandomForestClassifier` to predict bank product mappings based on specific features. The model is trained on data extracted from a SQL Server database and makes predictions for new product data. Keywords are also used to identify product types, such as Checking, Savings, Loan, CreditCard, and more.  
_Note:_ This version has numerous redacted sections due to sensitive information. For more information, email me @ geneparkmcs@gmail.com  

## Project Overview

This project:
- Extracts product mapping data from a SQL Server database.
- Uses keyword matching to identify different product types based on the `PROD_NAME` column along with different custom dimensions.
- Trains a fitted `RandomForestClassifier` to predict product mapping features.
- Predicts new data from a CSV file and writes the results back to a temporary SQL Server table.

## Requirements

The project requires the following Python libraries:

- `pandas`: Data manipulation and analysis.
- `scikit-learn`: Machine learning tools, including the `RandomForestClassifier` and data preprocessing.
- `pyodbc`: To connect to the SQL Server database.  

You can install all required libraries by using the following command:

```bash
pip install -r requirements.txt

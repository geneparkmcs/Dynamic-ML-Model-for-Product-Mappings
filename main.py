import warnings
from db_connection import connect_to_db
from data_processing import extract_data, add_keywords
from model_training import train_overfitted_model, evaluate_model
from prediction import load_new_data, predict_new_data, write_to_db
from sklearn.preprocessing import OneHotEncoder

UID = 'your_user'
PWD = 'your_password'
new_data_file = "path/to/file"
target_columns = ["CDA_LOB_DIRECT", "CDA_LOB_SEG", "CDA_PROD_GRP"]
categorical_columns = ["client_code", "LOB_DIRECT", "LOB_SEG", "PROD_GRP", "PROD_NAME",
                       "DIM1_NUMID", "DIM2_NUMID", "DIM3_NUMID", "DIM4_NUMID"]

product_keywords = {
    'Checking': ['Checking', 'Current'],
    'Savings': ['Savings', 'Money Market', 'Fixed Deposit', 'Term Deposit'],
    'Loan': ['Loan', 'Mortgage', 'Auto Loan', 'Business Loan', 'Student Loan', 'Credit Line', 'Home Equity'],
    'CreditCard': ['Credit Card', 'Debit Card', 'Prepaid Card'],
    'Investment': ['Investment', 'Mutual Fund', 'Brokerage', 'Treasury', 'Bond', 'ETF', 'Stock'],
    'Insurance': ['Insurance', 'Life Insurance', 'Health Insurance', 'Disability Insurance', 'Business Insurance'],
    'Service': ['Overdraft', 'Wire Transfer', 'Direct Deposit', 'Online Banking', 'Merchant Services'],
    'BusinessProduct': ['Commercial', 'Corporate', 'Merchant', 'Leasing', 'Treasury', 'Asset Management']}

def main():
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

    # Step 1: Connect to Database
    conn = connect_to_db(UID, PWD)
    if conn is None:
        return

    # Step 2: Extract Data from SQL
    df = extract_data(conn)

    # Step 3: Add Keyword Columns and Apply OneHotEncoder
    df = add_keywords(df)
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(df[categorical_columns])

    # Step 5: Train Overfitted Model
    model = train_overfitted_model(X_encoded, df[target_columns])

    # Step 6: Evaluate Model
    evaluate_model(model, X_encoded, df)

    # Step 7: Load and Predict on New Data
    new_data = load_new_data(new_data_file, product_keywords)
    result = predict_new_data(model, encoder, new_data)

    # Step 8: Write Predictions to SQL Database
    conn = connect_to_db(UID, PWD)
    if conn is not None:
        write_to_db(conn, result)

if __name__ == "__main__":
    main()

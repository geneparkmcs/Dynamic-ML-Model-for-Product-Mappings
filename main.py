import warnings
import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---- Configuration Section ----
client = ''
UID = ''
PWD = ''
new_data_file = "path/to/file"
target_columns = ["CDA_LOB_DIRECT", "CDA_LOB_SEG", "CDA_PROD_GRP"]
categorical_columns = ["client_code", "LOB_DIRECT", "LOB_SEG", "PROD_GRP", "PROD_NAME",
                       "DIM1_NUMID", "DIM2_NUMID", "DIM3_NUMID", "DIM4_NUMID"]

# List of product keywords for banks
product_keywords = {
    'Checking': ['Checking', 'Current'],
    'Savings': ['Savings', 'Money Market', 'Fixed Deposit', 'Term Deposit'],
    'Loan': ['Loan', 'Mortgage', 'Auto Loan', 'Business Loan', 'Student Loan', 'Credit Line', 'Home Equity'],
    'CreditCard': ['Credit Card', 'Debit Card', 'Prepaid Card'],
    'Investment': ['Investment', 'Mutual Fund', 'Brokerage', 'Treasury', 'Bond', 'ETF', 'Stock'],
    'Insurance': ['Insurance', 'Life Insurance', 'Health Insurance', 'Disability Insurance', 'Business Insurance'],
    'Service': ['Overdraft', 'Wire Transfer', 'Direct Deposit', 'Online Banking', 'Merchant Services'],
    'BusinessProduct': ['Commercial', 'Corporate', 'Merchant', 'Leasing', 'Treasury', 'Asset Management']
}

# ---- Database Connection Method ----
def connect_to_db():
    try:
        conn = pyodbc.connect(
            'DRIVER={SQL Server};'
            'SERVER=REDACTED;'
            'DATABASE=REDACTED;'
            'UID='+UID+';'
            'PWD='+PWD+';'
            'Trusted_Connection=yes'
        )
        print("Connection established")
        return conn
    except Exception as e:
        print("Couldn't Connect or Execute Query")
        print("Error:", e)
        return None

# ---- Data Extraction Method ----
def extract_data(conn):
    try:
        query = """
        SELECT client_code, LOB_DIRECT, LOB_SEG, PROD_GRP, PROD_NAME,
               DIM1_NUMID, DIM2_NUMID, DIM3_NUMID, DIM4_NUMID,
               CDA_LOB_DIRECT, CDA_LOB_SEG, CDA_PROD_GRP
        FROM LU_CDA_PRODMAP
        """
        df = pd.read_sql(query, conn)
        print("Data loaded successfully")
        return df
    finally:
        conn.close()
        print("Connection closed")

# ---- Add Keyword Columns ----
def add_keywords(df):
    # Add new columns for each product type based on keywords
    for product_type, keywords in product_keywords.items():
        df[product_type] = df['PROD_NAME'].str.contains('|'.join(keywords), case=False, na=False)
        categorical_columns.append(product_type)  

    return df

# ---- Model Training ----
def train_overfitted_model(X_encoded, y):
    clf_overfit = RandomForestClassifier(
        n_estimators=1000,       
        min_samples_split=2,     
        min_samples_leaf=1,      
        max_depth=None,         
        random_state=42
    )
    clf_overfit.fit(X_encoded, y)
    return clf_overfit

# ---- Model Evaluation ----
def evaluate_model(model, X_encoded, y):
    y_pred = model.predict(X_encoded)
    accuracy_direct = accuracy_score(y[target_columns[0]], y_pred[:, 0])
    accuracy_seg = accuracy_score(y[target_columns[1]], y_pred[:, 1])
    accuracy_prod_grp = accuracy_score(y[target_columns[2]], y_pred[:, 2])

    print(f"Overfitted Accuracy for {target_columns[0]}: {accuracy_direct}")
    print(f"Overfitted Accuracy for {target_columns[1]}: {accuracy_seg}")
    print(f"Overfitted Accuracy for {target_columns[2]}: {accuracy_prod_grp}")

# ---- Load New Data for Prediction ----
def load_new_data(file_path):
    new_data = pd.read_csv(file_path, header=0)

    # Add keyword columns to new data
    for product_type, keywords in product_keywords.items():
        new_data[product_type] = new_data['PROD_NAME'].str.contains('|'.join(keywords), case=False, na=False)

    return new_data

# ---- Predict New Data ----
def predict_new_data(model, encoder, new_data):
    new_data_encoded = encoder.transform(new_data[categorical_columns])
    new_data_predictions = model.predict(new_data_encoded)
    predictions_df = pd.DataFrame(new_data_predictions, columns=target_columns)
    result = pd.concat([new_data, predictions_df], axis=1)
    print("Predictions on new data:")
    print(result)
    return result

# ---- Writing to Temporary SQL Table ----
def write_to_db(conn, result):
    try:
        cursor = conn.cursor()
        print("Connection established for writing data")

        # Create a temporary table
        create_temp_table_query = """
        SELECT * INTO #temp_LU_CDA_PRODMAP FROM LU_CDA_PRODMAP WHERE 1=0
        """
        cursor.execute(create_temp_table_query)
        conn.commit()
        print("Temporary table created")

        # Insert the result DataFrame into the temporary table
        for index, row in result.iterrows():
            cursor.execute("""
                INSERT INTO #temp_LU_CDA_PRODMAP (client_code, LOB_DIRECT, LOB_SEG, PROD_GRP, PROD_NAME,
                                                  DIM1_NUMID, DIM2_NUMID, DIM3_NUMID, DIM4_NUMID,
                                                  CDA_LOB_DIRECT, CDA_LOB_SEG, CDA_PROD_GRP)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, row['client_code'], row['LOB_DIRECT'], row['LOB_SEG'], row['PROD_GRP'], row['PROD_NAME'],
               row['DIM1_NUMID'], row['DIM2_NUMID'], row['DIM3_NUMID'], row['DIM4_NUMID'],
               row['CDA_LOB_DIRECT'], row['CDA_LOB_SEG'], row['CDA_PROD_GRP'])

        conn.commit()
        print("Data inserted into temporary table")
    except Exception as e:
        print("Couldn't connect or insert data into the temp table")
        print("Error:", e)
    finally:
        conn.close()
        print("Connection closed after writing data")

# ---- Main Workflow Method ----
def main():
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

    # Step 1: Connect to Database
    conn = connect_to_db()
    if conn is None:
        return

    # Step 2: Extract Data from SQL
    df = extract_data(conn)

    # Step 3: Add Keyword Columns
    df = add_keywords(df)

    # Step 4: One-Hot Encoding of Categorical Variables
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(df[categorical_columns])

    # Step 5: Train Overfitted Model
    model = train_overfitted_model(X_encoded, df[target_columns])

    # Step 6: Evaluate Model
    evaluate_model(model, X_encoded, df)

    # Step 7: Load and Predict on New Data
    new_data = load_new_data(new_data_file)
    result = predict_new_data(model, encoder, new_data)

    # Step 8: Write Predictions to SQL Database
    conn = connect_to_db()
    if conn is not None:
        write_to_db(conn, result)

if __name__ == "__main__":
    main()

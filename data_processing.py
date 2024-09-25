import pandas as pd

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

categorical_columns = ["client_code", "LOB_DIRECT", "LOB_SEG", "PROD_GRP", "PROD_NAME",
                       "DIM1_NUMID", "DIM2_NUMID", "DIM3_NUMID", "DIM4_NUMID"]

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

import pandas as pd

target_columns = ["CDA_LOB_DIRECT", "CDA_LOB_SEG", "CDA_PROD_GRP"]
categorical_columns = ["client_code", "LOB_DIRECT", "LOB_SEG", "PROD_GRP", "PROD_NAME",
                       "DIM1_NUMID", "DIM2_NUMID", "DIM3_NUMID", "DIM4_NUMID"]

# ---- Load New Data for Prediction ----
def load_new_data(file_path, product_keywords):
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

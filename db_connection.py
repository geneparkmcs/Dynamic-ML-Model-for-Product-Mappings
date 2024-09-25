import pyodbc

# ---- Database Connection Method ----
def connect_to_db(UID, PWD):
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

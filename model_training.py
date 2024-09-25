from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

target_columns = ["CDA_LOB_DIRECT", "CDA_LOB_SEG", "CDA_PROD_GRP"]

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

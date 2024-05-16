import sqlite3 as sql
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

csv_file = 'heart.csv'
heartdb = sql.connect("heart.db")
df = pd.read_csv(csv_file, delimiter = ";")
print("CONNECTION SUCCESS")



cursor = heartdb.cursor()

create_table = '''
    CREATE TABLE IF NOT EXISTS heart (
        age INTEGER
        sex INTEGER
        cp INTEGER
        trestbps INTEGER
        chol INTEGER
        fbs INTEGER
        restecg INTEGER
        thalach INTEGER
        exang INTEGER
        oldpeak REAL
        slope INTEGER
        ca INTEGER
        thal INTEGER
        target INTEGER
        )
    '''

cursor.execute(create_table)
print("Table Created")


df.to_sql("heart", heartdb, if_exists='replace', index=False)
print("DataFrame stored in SQLite database successfully!")

heartdb.commit()
print("COMMITED")


df = pd.read_sql_query("SELECT * from heart", heartdb)



# Ensure the target variable is discrete
df['target'] = df['target'].astype(int)

# Display unique values of the target variable
print(df['target'].unique())

print("Cleaning Data")
df.dropna(inplace = True)

for x in df.index:
    df.loc[x, "oldpeak"] / 1.0
    
df.drop_duplicates(inplace = True)
print("Cleaning Complete")



# Handle missing values
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    df[column].fillna(df[column].median(), inplace=True)

for column in df.select_dtypes(include=['object']).columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Encode categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Scale numeric features
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Define the target variable and feature variables
X = df.drop('target', axis=1)
y = df['target'].astype(int)  # Ensure target variable is integer

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Initialize the models
log_reg = LogisticRegression(max_iter=10000, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='linear', random_state=42, probability=True)

# Fit the models
log_reg.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Predict on the test set
y_pred_log_reg = log_reg.predict(X_test)
y_pred_random_forest = random_forest.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Evaluate the models
print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg)}")
print(classification_report(y_test, y_pred_log_reg))

print("\nRandom Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_random_forest)}")
print(classification_report(y_test, y_pred_random_forest))

print("\nSVM Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print(classification_report(y_test, y_pred_svm))

# Determine which model performs the best
models = {'Logistic Regression': log_reg, 'Random Forest': random_forest, 'SVM': svm}
accuracies = {'Logistic Regression': accuracy_score(y_test, y_pred_log_reg),
              'Random Forest': accuracy_score(y_test, y_pred_random_forest),
              'SVM': accuracy_score(y_test, y_pred_svm)}

best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name}")

# Save the best model to disk
joblib_file = f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl"
joblib.dump(best_model, joblib_file)
print(f"Best model saved as {joblib_file}")

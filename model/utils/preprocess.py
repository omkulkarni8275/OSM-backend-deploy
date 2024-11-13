import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib  # Import joblib to save the models

# Load the dataset
df = pd.read_csv(r"D:\Acad\TY\Labs\MP-1\updated_ocd_dataset_with_diagnosis.csv")

# Fill missing values
df.fillna({
    'Obsession Type': 'None',  # Assuming 'None' is a valid category for this column
    'Compulsion Type': 'None',
    'Depression Diagnosis': 0,  # Assuming 0 for no depression diagnosis
    'Anxiety Diagnosis': 0  # Assuming 0 for no anxiety diagnosis
}, inplace=True)

# Function to determine OCD seriousness
def get_ocd_seriousness(row):
    ocd_severity = ''
    ocd_percentage = 0
    
    # Adjust severity based on multiple factors
    if row['Duration of Symptoms (months)'] >= 24 and row['Obsession Type'] != 'None' and row['Compulsion Type'] != 'None' and row['Depression Diagnosis'] == 1 and row['Anxiety Diagnosis'] == 1:
        ocd_severity = 'Extreme'
        ocd_percentage = 90
    elif row['Duration of Symptoms (months)'] >= 18 and row['Obsession Type'] != 'None' and row['Compulsion Type'] != 'None':
        ocd_severity = 'Severe'
        ocd_percentage = 80
    elif row['Duration of Symptoms (months)'] >= 12 and (row['Obsession Type'] != 'None' or row['Compulsion Type'] != 'None') and (row['Depression Diagnosis'] == 1 or row['Anxiety Diagnosis'] == 1):
        ocd_severity = 'Moderate'
        ocd_percentage = 65
    elif row['Duration of Symptoms (months)'] >= 6 and (row['Obsession Type'] != 'None' or row['Compulsion Type'] != 'None'):
        ocd_severity = 'Mild'
        ocd_percentage = 40
    else:
        ocd_severity = 'Minimal'
        ocd_percentage = 20

    # Adjust severity further based on age and other patterns
    if row['Age'] < 30 and row['Duration of Symptoms (months)'] >= 12:
        ocd_percentage += 10
    elif row['Age'] > 60 and row['Duration of Symptoms (months)'] < 12:
        ocd_percentage -= 5
    
    return pd.Series([ocd_severity, ocd_percentage])

# Apply the function to calculate 'OCD Severity' and 'OCD Percentage'
df[['OCD Severity', 'OCD Percentage']] = df.apply(get_ocd_seriousness, axis=1)

# Save the updated dataframe
df.to_csv(r"D:\Acad\TY\Labs\MP-1\updated_ocd_dataset_with_diagnosis.csv", index=False)

# Step 2: Label Encoding for the severity column
label_encoder = LabelEncoder()
df['OCD Severity Label'] = label_encoder.fit_transform(df['OCD Severity'])

# Step 3: Prepare features and labels for the model
# Convert 'Yes'/'No' in diagnosis columns to 1/0 for Depression and Anxiety Diagnosis
df['Depression Diagnosis'] = df['Depression Diagnosis'].replace({'Yes': 1, 'No': 0}).astype(int)
df['Anxiety Diagnosis'] = df['Anxiety Diagnosis'].replace({'Yes': 1, 'No': 0}).astype(int)

# Handle categorical variables 'Obsession Type' and 'Compulsion Type' using one-hot encoding
X = df[['Age', 'Duration of Symptoms (months)', 'Obsession Type', 'Compulsion Type', 'Depression Diagnosis', 'Anxiety Diagnosis']]

# Encode categorical columns with one-hot encoding
X = pd.get_dummies(X, columns=['Obsession Type', 'Compulsion Type'])

# Prepare the target variables
y_percentage = df['OCD Percentage']
y_severity = df['OCD Severity Label']

# Step 4: Split data into training and testing sets
X_train, X_test, y_train_percentage, y_test_percentage, y_train_severity, y_test_severity = train_test_split(
    X, y_percentage, y_severity, test_size=0.2, random_state=42
)

# Step 5: Train Linear Regression Model for OCD Percentage (Regression)
regressor = LinearRegression()
regressor.fit(X_train, y_train_percentage)

# Predict and evaluate for percentage
y_pred_percentage = regressor.predict(X_test)

# Step 6: Train Decision Tree for OCD Severity (Classification)
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train_severity)

# Predict and evaluate for severity
y_pred_severity = classifier.predict(X_test)
accuracy = accuracy_score(y_test_severity, y_pred_severity)
print(f'Accuracy for OCD Severity Classification: {accuracy}')

# Step 7: Plot the decision tree for OCD Severity
plt.figure(figsize=(12, 8))
plot_tree(classifier, filled=True, feature_names=X.columns, class_names=label_encoder.classes_, fontsize=10)
plt.title('Decision Tree for OCD Severity Classification')
plt.show()

# Step 8: Graph the predicted OCD percentage for the test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test_percentage, y_pred_percentage, color='blue')
plt.plot([y_test_percentage.min(), y_test_percentage.max()], [y_test_percentage.min(), y_test_percentage.max()], color='red', linestyle='--')
plt.xlabel('True OCD Percentage')
plt.ylabel('Predicted OCD Percentage')
plt.title('True vs Predicted OCD Percentage')
plt.show()

# Step 9: Example Prediction for a New Data Point
example_data = {
    'Age': 25, 
    'Duration of Symptoms (months)': 14,
    'Obsession Type': 'Contamination', 
    'Compulsion Type': 'Checking', 
    'Depression Diagnosis': 1,
    'Anxiety Diagnosis': 1
}

# Convert to DataFrame
example_df = pd.DataFrame([example_data])

# Prepare the example data just like in the code
example_df = pd.get_dummies(example_df, columns=['Obsession Type', 'Compulsion Type'])

# Align columns with X (ensure all dummy variables match)
example_df = example_df.reindex(columns=X.columns, fill_value=0)

# Predict OCD Percentage
predicted_percentage = regressor.predict(example_df)
print(f'Predicted OCD Percentage: {predicted_percentage[0]}')

# Predict OCD Severity
predicted_severity = classifier.predict(example_df)
predicted_severity_label = label_encoder.inverse_transform(predicted_severity)
print(f'Predicted OCD Severity: {predicted_severity_label[0]}')

# Save the trained models using joblib
joblib.dump(regressor, 'ocd_percentage_model.joblib')
joblib.dump(classifier, 'ocd_severity_model.joblib')
joblib.dump(label_encoder, 'ocd_severity_label_encoder.joblib')

print("Models saved successfully!")

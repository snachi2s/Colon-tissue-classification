# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset into a pandas DataFrame
# Replace 'your_data.csv' with the actual file name or provide your DataFrame
data = pd.read_csv('your_data.csv')

# Assume 'label' is the column you want to predict
X = data.drop('label', axis=1)  # Features
y = data['label']  # Labels

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (recommended for SVM, optional for Random Forest)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM Model
svm_model = SVC(kernel='linear', C=1)  # You can adjust the parameters based on your needs
svm_model.fit(X_train, y_train)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the parameters
rf_model.fit(X_train, y_train)

# Predictions
svm_predictions = svm_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Evaluate the models
svm_accuracy = accuracy_score(y_test, svm_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("SVM Accuracy:", svm_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

# Additional evaluation metrics (optional)
print("\nSVM Classification Report:\n", classification_report(y_test, svm_predictions))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_predictions))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import numpy as np
data = pd.read_csv('parkinsons.data')

X = data.drop(columns=['name', 'status']) 
y = data['status'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5, n_jobs=1) 
model.fit(X_train, y_train)

joblib.dump(model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

new_data = np.array([[0.029, 0.047, -0.008, 0.019, 0.002, -0.015, 0.004, 0.021,
                      0.047, -0.041, -0.006, 0.030, 0.029, -0.029, -0.036, 0.018,
                      0.014, -0.013, 0.004, 0.004, 0.010, 0.011]])  # Ensure exactly 22 features
new_data_df = pd.DataFrame(new_data, columns=X.columns)
new_data_scaled = scaler.transform(new_data_df)
prediction = model.predict(new_data_scaled)
print("\nPredicted class for the new sample:", prediction)
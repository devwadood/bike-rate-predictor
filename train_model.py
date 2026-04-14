import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

print("Step 1: Naya Data Load kar rahe hain...")
df = pd.read_csv('bikes_dataset.csv')

df['bike_age'] = 2026 - df['model_year']

X = df[['brand', 'engine_capacity', 'bike_age', 'mileage', 'condition_score']]
y = df['price']

print("Step 2: Brands ko encode kar rahe hain...")
X = pd.get_dummies(X, columns=['brand'])
model_columns = X.columns.tolist()

print("Step 3: Training shuru...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'bike_linear_model.pkl')
joblib.dump(model_columns, 'model_columns.pkl')

print("Mubarak ho! Linear model with CC and Future Prediction save ho gaya hai.")
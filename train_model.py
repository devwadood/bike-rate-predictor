"""Train and save the bike price Linear Regression model."""

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():
    """Load the dataset, train the model, and save the fitted artifacts."""

    print("Step 1: Naya Data Load kar rahe hain...")
    df = pd.read_csv('bikes_dataset.csv')

    df['bike_age'] = 2026 - df['model_year']

    features = df[['brand', 'engine_capacity', 'bike_age', 'mileage', 'condition_score']]
    target = df['price']

    print("Step 2: Brands ko encode kar rahe hain...")
    features = pd.get_dummies(features, columns=['brand'])
    model_columns = features.columns.tolist()

    print("Step 3: Training shuru...")
    features_train, _, target_train, _ = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(features_train, target_train)

    joblib.dump(model, 'bike_linear_model.pkl')
    joblib.dump(model_columns, 'model_columns.pkl')

    print("Mubarak ho! Linear model with CC and Future Prediction save ho gaya hai.")


if __name__ == "__main__":
    main()

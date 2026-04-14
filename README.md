# Bike Price Predictor API

This project predicts motorcycle resale prices using a Linear Regression model. It includes a training script, a sample dataset, and a FastAPI server for making predictions through an API.

## Features

- Trains a `LinearRegression` model with scikit-learn.
- Uses bike brand, engine capacity, bike age, mileage, and condition score as model inputs.
- Encodes bike brands with one-hot encoding using `pandas.get_dummies`.
- Saves the trained model with `joblib`.
- Provides a FastAPI `/predict` endpoint.
- Supports optional future-year prediction by recalculating the bike age for the selected year.

## Project Structure

```text
.
|-- bikes_dataset.csv       # Training dataset
|-- train_model.py          # Trains and saves the Linear Regression model
|-- main.py                 # FastAPI app for price prediction
|-- requirements.txt        # Python dependencies
|-- bike_linear_model.pkl   # Saved trained model
`-- model_columns.pkl       # Saved input column order used during training
```

## Dataset Format

The model expects `bikes_dataset.csv` to contain these columns:

```csv
brand,model_year,engine_capacity,mileage,condition_score,price
```

Column meanings:

- `brand`: Bike brand, such as `Honda`, `Yamaha`, or `Suzuki`.
- `model_year`: Manufacturing/model year of the bike.
- `engine_capacity`: Engine size in CC, such as `70`, `100`, `125`, or `150`.
- `mileage`: Total kilometers driven.
- `condition_score`: Bike condition rating.
- `price`: Actual resale price in rupees. This is the value the model learns to predict.

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python train_model.py
```

The training script:

1. Loads `bikes_dataset.csv`.
2. Creates a `bike_age` column from `model_year`.
3. Uses these input features:
   - `brand`
   - `engine_capacity`
   - `bike_age`
   - `mileage`
   - `condition_score`
4. Trains a `LinearRegression` model.
5. Saves:
   - `bike_linear_model.pkl`
   - `model_columns.pkl`

Note: the current training script calculates `bike_age` using the year `2026`.

## Run the API

Start the FastAPI server:

```bash
uvicorn main:app --reload --port 8002
```

Open the API documentation in your browser:

```text
http://127.0.0.1:8002/docs
```

## API Usage

### Health Check

```http
GET /
```

Example response:

```json
{
  "message": "Advanced Bike API is Running!"
}
```

### Predict Bike Price

```http
POST /predict
```

Example request:

```json
{
  "brand": "Honda",
  "model_year": 2024,
  "engine_capacity": 125,
  "mileage": 5000,
  "condition_score": 9
}
```

Example request with future-year prediction:

```json
{
  "brand": "Honda",
  "model_year": 2024,
  "engine_capacity": 125,
  "mileage": 5000,
  "condition_score": 9,
  "target_year": 2027
}
```

Example response:

```json
{
  "status": "success",
  "predicted_price_rs": 197553,
  "calculated_for_year": 2027,
  "calculated_bike_age": 3,
  "bike_details": {
    "brand": "Honda",
    "model_year": 2024,
    "engine_capacity": 125,
    "mileage": 5000,
    "condition_score": 9,
    "target_year": 2027
  }
}
```

## Important Notes

- The API loads `bike_linear_model.pkl` and `model_columns.pkl` when `main.py` starts.
- Run `python train_model.py` again whenever you update `bikes_dataset.csv`.
- If a requested `target_year` is earlier than the bike `model_year`, the API returns a validation error.
- Linear Regression can produce negative values for unusual inputs, so the API converts negative predictions to `0`.
- Prediction quality depends heavily on the size and quality of `bikes_dataset.csv`.

## Dependencies

Main libraries used:

- `pandas`
- `scikit-learn`
- `joblib`
- `fastapi`
- `uvicorn`

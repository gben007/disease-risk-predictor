# Disease Risk Predictor

A Streamlit web app that predicts whether a patient is at low or high disease risk using a trained `DecisionTreeClassifier`.

## Features

- Streamlit-based interface for entering patient health and lifestyle data
- Predicts disease risk with a pre-trained model stored in `disease_risk_model.pkl`
- Shows model confidence when `predict_proba` is available
- Lets users inspect the processed input row before export
- Downloads prediction results as a CSV file

## Project Files

- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies
- `disease_risk_model.pkl` - trained machine learning model
- `health_lifestyle_dataset.csv` - sample dataset used for training or reference

## Input Fields

The app collects the following values:

- `age`
- `gender`
- `bmi`
- `daily_steps`
- `sleep_hours`
- `water_intake_l`
- `calories_consumed`
- `smoker`
- `alcohol`
- `resting_hr`
- `systolic_bp`
- `diastolic_bp`
- `cholesterol`
- `family_history`

Optional:

- `id` if the trained model expects it as an input feature

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/gben007/disease-risk-predictor.git
cd disease-risk-predictor
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

After that, open the local URL shown in the terminal, usually:

`http://localhost:8501`

## How It Works

1. The app loads the pickled model from `disease_risk_model.pkl`.
2. User inputs are converted into a single-row pandas DataFrame.
3. Binary values are encoded inside the app:
   - `Male` -> `1`, `Female` -> `0`
   - `Yes` -> `1`, `No` -> `0`
4. The model predicts the disease risk label.
5. If supported by the model, the app also shows a confidence score.

## Important Notes

- The current model was trained with the `id` column included as a feature. In the app, turn on the sidebar option `Model was trained with 'id' column` when using this model.
- The categorical encoding used in the app must match the encoding used during training.
- If prediction fails because of column mismatch, verify that the model expects the same feature set and order sent by `app.py`.
- Pickled scikit-learn models can be sensitive to library version differences. If you see model loading warnings, try using the same scikit-learn version that was used during training.

## Dependencies

- Streamlit
- pandas
- scikit-learn

## Future Improvements

- Add a training notebook or script to reproduce the model
- Add model evaluation metrics and confusion matrix
- Add tests for input preparation and prediction flow
- Deploy the app to Streamlit Community Cloud or another hosting platform

## Disclaimer

This project is for educational and demonstration purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

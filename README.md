

# Bank Transaction Fraud Detection

A machine learning project to detect fraudulent bank transactions using a Random Forest classifier. The project includes data preprocessing, model training with class balancing, prediction, and an interactive Streamlit dashboard for visualization and prediction.

## Features
- **Fraud Detection Model:** Trained with SMOTE for class balancing and feature engineering for improved accuracy.
- **Streamlit Dashboard:** User-friendly interface for predictions, EDA, and model metrics.
- **EDA Visualizations:** Explore fraud distribution, transaction types, hourly trends, and feature correlations.
- **Feature Importance:** See which features contribute most to fraud prediction.

## Project Structure
```
├── data/
│   └── transactions.csv
├── models/
│   ├── fraud_model_rf.joblib
│   └── metrics.json
├── src/
│   ├── app.py           # Streamlit UI
│   ├── train.py         # Model training
│   ├── predict.py       # Prediction logic
│   ├── data_preprocessing.py # Preprocessing & feature engineering
│   ├── eda.py           # EDA visualizations
├── README.md
```

## How to Run
1. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
2. **Train the model:**
  ```bash
  python src/train.py
  ```
3. **Start the Streamlit app:**
  ```bash
  streamlit run src/app.py
  ```

## Usage
- Use the dashboard to input transaction details and get fraud predictions.
- Explore EDA tabs for insights into fraud patterns.
- View model metrics and feature importance for transparency.

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, imbalanced-learn, joblib, streamlit, matplotlib, seaborn

## License
MIT

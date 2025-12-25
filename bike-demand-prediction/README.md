# Bike Demand Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

A machine learning pipeline for predicting bike-sharing demand using polynomial regression models with custom feature engineering. This project demonstrates comparative analysis of multiple regression models, from simple linear to complex polynomial with interaction terms.

## Project Highlights

- **Custom Feature Engineering**: Implemented temporal features (hour, day, month) from datetime
- **Multiple Model Comparison**: Evaluated 5+ regression variants including custom polynomial transformers
- **Automated Pipeline**: End-to-end ML pipeline from data preprocessing to visualization
- **Performance Metrics**: Comprehensive evaluation using R², MAE, MSE, and RMSE
- **Visual Analytics**: Automated generation of actual vs. predicted plots for model comparison

## Results

The project compares multiple regression approaches:
- **Linear Regression**: Baseline model (R² = 0.696)
- **Polynomial Features (Degree 2-4)**: Custom power transformations without interactions (R² = 0.697-0.703)
- **Quadratic with Interactions**: Full polynomial feature expansion using scikit-learn (**R² = 0.704**)

**Best Model**: Quadratic with interaction terms achieved **70.4% R² score**, explaining 70% of variance in bike demand.

*Detailed metrics are automatically saved to `results/best_model.txt` and `results/metrics_table.csv`*

## Tech Stack

- **Python 3.8+**
- **NumPy & Pandas**: Data manipulation
- **Scikit-learn**: ML pipeline, preprocessing, and models
- **Matplotlib**: Visualization
- **Custom Transformers**: PowerFeatures class for flexible polynomial expansion

## Project Structure

```
bike-demand-prediction/
├── src/
│   ├── run_all.py                  # Main execution script
│   ├── preprocess.py               # Data loading and feature engineering
│   ├── model_definitions.py        # Custom transformers and model pipeline
│   ├── train_and_evaluate.py       # Model training and metric calculation
│   └── plot_results.py             # Visualization generation
├── data/
│   └── train.csv                   # Bike sharing dataset
├── results/                        # Auto-generated outputs
│   ├── best_model.txt              # Best performing model summary
│   ├── metrics_table.csv           # Detailed metrics comparison
│   └── plots/                      # Actual vs. predicted visualizations
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd bike-demand-prediction

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
cd src
python run_all.py
```

This will:
1. Load and preprocess the dataset
2. Train all 5 regression models
3. Evaluate performance metrics
4. Generate comparison visualizations
5. Save results to `results/` directory

## Features & Methodology

### Feature Engineering
- **Temporal Features**: Extracted hour, day of week, month, year from datetime
- **Numeric Features**: Temperature, feels-like temperature, humidity, wind speed
- **Categorical Features**: Season, holiday, working day, weather conditions

### Model Architecture
1. **Preprocessing Pipeline**:
   - StandardScaler for numeric features
   - OneHotEncoder for categorical features
   - Custom PowerFeatures transformer for polynomial expansion

2. **Model Variants**:
   - Linear regression (baseline)
   - Polynomial degrees 2-4 without interactions (custom)
   - Quadratic with full interaction terms (scikit-learn)

### Evaluation Metrics
- **R² Score**: Model goodness of fit
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error

## Visualizations

The project automatically generates:
- Actual vs. Predicted scatter plots for each model
- Visual comparison of model performance
- Saved as high-quality PNG images in `results/plots/`

## Key Learnings

- **Custom Transformers**: Implemented sklearn-compatible PowerFeatures transformer
- **Pipeline Design**: Built modular, reusable ML pipeline components
- **Model Comparison**: Systematic evaluation of model complexity vs. performance
- **Feature Engineering**: Demonstrated impact of temporal feature extraction

## Customization

To experiment with different configurations:

1. **Add new features**: Modify `preprocess.py`
2. **Try new models**: Add to `model_definitions.py`
3. **Change evaluation metrics**: Update `train_and_evaluate.py`
4. **Customize plots**: Modify `plot_results.py`

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Contributing

Contributions, issues, and feature requests are welcome!

## Contact

Feel free to reach out for any questions or collaboration opportunities.

---

**Note**: This project was developed as part of a machine learning portfolio to demonstrate end-to-end ML pipeline development, custom transformer implementation, and comparative model analysis.

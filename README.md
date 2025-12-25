# Machine Learning & Optimization Portfolio

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A portfolio showcasing machine learning and optimization projects, demonstrating proficiency in data science, algorithm implementation, and end-to-end ML pipeline development.

## Table of Contents

- [Projects](#projects)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Key Skills Demonstrated](#key-skills-demonstrated)
- [License](#license)

## Projects

### 1. [Bike Demand Prediction System](./bike-demand-prediction)

**Featured Project** - A comprehensive machine learning pipeline for predicting bike-sharing demand using advanced regression techniques.

**Highlights:**
- **Custom ML Pipeline**: Implemented scikit-learn compatible transformers for flexible feature engineering
- **Comparative Analysis**: Evaluated 5+ regression models from linear to 4th-degree polynomial
- **Automated Visualization**: Generated performance metrics and actual vs. predicted plots
- **Feature Engineering**: Extracted temporal patterns (hour, day, month) and weather correlations

**Tech Stack**: Python, Scikit-learn, Pandas, NumPy, Matplotlib

**Key Results**:
- **Achieved 70.4% R² score** on bike-sharing demand prediction using quadratic regression with interaction terms
- Built modular preprocessing pipeline with StandardScaler and OneHotEncoder
- Implemented custom PowerFeatures transformer for polynomial expansion
- Compared 5 regression models systematically, with best model reducing MSE by 2.7% over linear baseline
- Automated result logging and visualization generation with actual vs. predicted plots

[View Full Documentation →](./bike-demand-prediction/README.md)

---

### 2. [Constrained Optimization Solver](./constrained-optimization)

Implementation of Lagrangian relaxation and dual gradient ascent algorithms for solving constrained optimization problems.

**Highlights:**
- **Algorithm Implementation**: Dual gradient ascent from first principles
- **Convergence Analysis**: Visualization of optimization trajectory
- **Modular Design**: Reusable components for problem definition and solving

**Tech Stack**: Python, NumPy, Matplotlib

[View Documentation →](./constrained-optimization/README.md)

## Tech Stack

### Languages & Libraries
- **Python 3.8+**: Primary programming language
- **NumPy & Pandas**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning models and pipelines
- **Matplotlib**: Data visualization

### Skills & Concepts
- Machine Learning (Regression, Feature Engineering)
- Algorithm Implementation (Optimization, Gradient Methods)
- Software Engineering (Modular Design, OOP)
- Data Analysis & Visualization
- Pipeline Development & Automation

## Quick Start

### Prerequisites
```bash
python --version  # Python 3.8 or higher required
```

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd optimisation-project

# Install dependencies for bike demand prediction
cd bike-demand-prediction
pip install -r requirements.txt

# Run the project
cd src
python run_all.py
```

Results will be generated in the `results/` directory with:
- Performance metrics comparison
- Model evaluation results
- Visualization plots

## Key Skills Demonstrated

### Machine Learning & Data Science
- End-to-end ML pipeline development
- Custom scikit-learn transformer implementation
- Feature engineering and selection
- Model evaluation and comparison
- Data preprocessing and normalization

### Software Engineering
- Modular, reusable code architecture
- Object-oriented design (custom transformers)
- Clean code principles and documentation
- Automated testing and result generation
- Version control best practices

### Algorithms & Optimization
- Polynomial regression and regularization
- Gradient-based optimization methods
- Lagrangian relaxation techniques
- Convergence analysis

### Tools & Technologies
- Python ecosystem (NumPy, Pandas, Scikit-learn)
- Data visualization (Matplotlib)
- Git version control
- Documentation and technical writing

## Project Statistics

| Metric | Value |
|--------|-------|
| **Programming Languages** | Python |
| **ML Models Implemented** | 5+ |
| **Custom Transformers** | 2 |
| **Evaluation Metrics** | 4 (R², MAE, MSE, RMSE) |
| **Lines of Code** | ~500+ |
| **Test Coverage** | Automated result validation |

## Repository Structure

```
optimisation-project/
├── bike-demand-prediction/     # Main ML project
│   ├── src/                    # Source code
│   ├── data/                   # Dataset
│   ├── results/                # Generated outputs
│   └── README.md               # Project documentation
├── constrained-optimization/   # Optimization algorithms
│   ├── src/                    # Algorithm implementations
│   ├── results/                # Solutions and plots
│   └── README.md               # Project documentation
├── docs/                       # Technical analysis reports
│   ├── bike-demand-analysis.pdf
│   └── optimization-analysis.pdf
├── assets/                     # Images and media
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## Learning Outcomes

This portfolio demonstrates:
- **Practical ML Application**: Real-world regression problem with bike-sharing data
- **Custom Implementation**: Built sklearn-compatible transformers from scratch
- **Systematic Evaluation**: Compared multiple model architectures scientifically
- **Algorithm Mastery**: Implemented optimization algorithms with mathematical foundation
- **Professional Development**: Industry-standard code structure and documentation

## Future Enhancements

Planned improvements:
- [ ] Add interactive web dashboard (Streamlit/Gradio)
- [ ] Implement unit tests with pytest
- [ ] Add CI/CD pipeline with GitHub Actions
- [ ] Deploy model as REST API
- [ ] Add time series forecasting models (ARIMA, LSTM)
- [ ] Expand to other datasets and domains

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## Contact

- GitHub: [@xerlixx](https://github.com/xerlixx)

---

<div align="center">

**Star this repository if you find it helpful!**

Made with Python

</div>

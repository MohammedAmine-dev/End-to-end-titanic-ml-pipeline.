# 🚢 Titanic Survival Prediction - Complete ML Pipeline

A comprehensive machine learning project demonstrating the complete workflow from data exploration to model deployment. This project showcases best practices in ML engineering including modular code architecture, reproducible pipelines, and proper documentation.

##  Project Overview

This project builds a predictive model to determine passenger survival on the Titanic using historical data. It demonstrates a complete ML workflow with:
-  Exploratory Data Analysis (EDA)
-  Data cleaning and preprocessing
-  Advanced feature engineering
-  Multiple model training and evaluation (7 models)
-  Hyperparameter tuning and optimization
-  Production-ready pipeline architecture


##  Learning Objectives

- Master data cleaning and preprocessing techniques
- Implement feature engineering strategies (title extraction, family size, binning)
- Build and compare multiple ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, SVM, Naive Bayes)
- Create reusable, modular code architecture
- Develop production-ready ML pipelines with proper configuration management
- Document findings and create reproducible workflows

## 📁 Project Structure

```
titanic-ml-pipeline/
├── data/                              # Raw and processed data
│   ├── train.csv                      # Original training data (891 samples)
│   ├── test.csv                       # Original test data
│   ├── train_cleaned.csv              # After Phase 2 (cleaning)
│   ├── train_engineered.csv           # After Phase 3 (feature engineering)
│   ├── gender_submission.csv          # Submission template
│   └── download_instructions.txt      # Dataset download guide
│
├── notebooks/                         # Jupyter notebooks for exploration & analysis
│   ├── 01_data_exploration.ipynb      # EDA, data profiling, distributions
│   ├── 02_data_cleaning.ipynb         # Handling missing values, outliers
│   ├── 03_feature_engineering.ipynb   # Feature creation, encoding, scaling
│   └── 04_model_training.ipynb        # Model training, comparison, tuning
│
├── src/                               # Production pipeline modules (Python)
│   ├── __init__.py
│   ├── config.py                      # Configuration management (PipelineConfig)
│   ├── data.py                        # Data loading utilities
│   ├── features.py                    # Feature preparation & engineering
│   ├── train.py                       # Model definitions & training helpers
│   ├── evaluate.py                    # Model evaluation & best model selection
│   ├── pipeline.py                    # Main orchestrator (end-to-end workflow)
│   └── predict.py                     # Prediction utilities for new data
│
├── results/                           # Model outputs and artifacts
│   └── titanic_best_model.pkl         # Serialized best model
│
├── example_prediction.py              # Demo script showing model predictions
├── .gitignore                         # Git ignore configuration
├── requirements.txt                   # Project dependencies
├── README.md                          # This file
└── .git/                              # Git repository
```

##  Quick Start

### Prerequisites
- Python 3.10 or higher
- pip or conda

### Installation

1. **Clone this repository**
```bash
git clone https://github.com/yourusername/titanic-ml-pipeline.git
cd titanic-ml-pipeline
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
- Follow instructions in `data/download_instructions.txt`
- Or download from [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- Place `train.csv` and `test.csv` in the `data/` directory

### Usage

**Option 1: Interactive Exploration** (Recommended for learning)
```bash
# Start Jupyter and explore each phase
jupyter notebook
# Open notebooks/01_data_exploration.ipynb through 04_model_training.ipynb
```

**Option 2: Run the Complete Pipeline** (Production)
```bash
python src/pipeline.py
```
This will:
- Load cleaned and engineered data
- Train all 7 models
- Evaluate and compare performance
- Save the best model to `results/titanic_best_model.pkl`

**Option 3: Make Predictions with the Trained Model**
```bash
python example_prediction.py
```
This will:
- Load the saved model
- Make predictions for sample passengers
- Show survival probabilities
- Demonstrate how to predict for your own inputs

##  Dataset Information

| Aspect | Details |
|--------|---------|
| **Source** | Kaggle Titanic Competition |
| **Train Size** | 891 passengers |
| **Test Size** | 418 passengers |
| **Features** | 12 original (PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked) |
| **Target** | Survived (Binary: 0 = Did not survive, 1 = Survived) |
| **Missing Data** | Age (~20%), Cabin (~77%), Embarked (~0.2%) |

##  Machine Learning Models

The pipeline trains and evaluates **7 different models**:

| Model | Type | Key Features |
|-------|------|--------------|
| **Logistic Regression** | Linear | Fast baseline, interpretable |
| **Decision Tree** | Tree-based | Simple, prone to overfitting |
| **Random Forest** | Ensemble | Robust, good generalization |
| **XGBoost** | Gradient Boosting | Highest performance, tuned hyperparameters |
| **K-Nearest Neighbors** | Distance-based | Simple, requires scaling |
| **Support Vector Machine** | Distance-based | Good for high-dimensional data |
| **Naive Bayes** | Probabilistic | Fast, works well with discrete features |

**Best Model Selection:** The pipeline automatically selects the model with the highest accuracy on the test set.

##  Key Technologies

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Programming language |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **Scikit-learn 1.3+** | Machine learning framework |
| **XGBoost 2.0+** | Gradient boosting |
| **Matplotlib & Seaborn** | Data visualization |
| **Jupyter** | Interactive notebooks |

## Project Phases

### Phase 1: Data Exploration (`01_data_exploration.ipynb`)
- Load and inspect data
- Analyze distributions and relationships
- Identify missing values
- Create preliminary visualizations

### Phase 2: Data Cleaning (`02_data_cleaning.ipynb`)
- Handle missing values (imputation strategy)
- Remove outliers
- Fix data inconsistencies
- Output: `train_cleaned.csv`

### Phase 3: Feature Engineering (`03_feature_engineering.ipynb`)
- Extract title from names
- Create family size features
- Bin continuous variables (Age, Fare)
- One-hot encode categorical features
- Output: `train_engineered.csv`

### Phase 4: Model Training (`04_model_training.ipynb`)
- Train 7 ML models
- Evaluate with accuracy & classification report
- Hyperparameter tuning (Random Forest, XGBoost)
- Feature importance analysis
- Before/after tuning comparison

### Phase 5: Production Pipeline (`src/pipeline.py`)
- Modular, reusable code architecture
- Configuration-driven approach
- Automatic best model selection
- Model serialization (pickle)
- Ready for deployment

##  Code Architecture

The `src/` directory follows a modular design:

- **config.py**: Centralized configuration via `PipelineConfig` dataclass
- **data.py**: Data loading utilities
- **features.py**: Feature preparation and engineering
- **train.py**: Model definitions with tuned hyperparameters + training helpers
- **evaluate.py**: Model evaluation and best model selection
- **pipeline.py**: Orchestrator that ties everything together
- **predict.py**: Prediction utilities for making survival predictions on new data

This architecture ensures:
-  Reusability across projects
-  Easy testing and debugging
-  Clear separation of concerns
-  Simple configuration management



##  Making Predictions

The project includes two prediction modules:

### 1. **predict.py** (Production Module)
```python
from src.predict import load_model, predict_single

# Load trained model
model = load_model("results/titanic_best_model.pkl")

# Define passenger using ACTUAL engineered features
passenger = {
    'Unnamed: 0': 0,               # Index
    'Pclass': 1,                   # Passenger class (1-3)
    'Sex': 0,                      # Gender (0=Female, 1=Male)
    'HasCabin': 1,                 # Has cabin info (0/1)
    'FamilySize': 2,               # Number of family members
    'IsAlone': 0,                  # Traveling alone (0/1)
    'Title_Master': 0,             # Title encoding
    'Title_Miss': 0,
    'Title_Mr': 0,
    'Title_Mrs': 1,
    'Title_Rare': 0,
    'AgeGroup_Child': 0,           # Age group encoding
    'AgeGroup_Teen': 0,
    'AgeGroup_Adult': 1,
    'AgeGroup_Senior': 0,
    'FareGroup_Low': 0,            # Fare group encoding
    'FareGroup_Medium': 0,
    'FareGroup_High': 1,
    'Embarked_C': 1,               # Embarkation port (C/Q/S)
    'Embarked_Q': 0,
    'Embarked_S': 0,
}

# Make prediction
prediction = predict_single(model, passenger)


print(f"Survived: {prediction}")       # 0 or 1

```

### 2. **example_prediction.py** (Demo Script)
Interactive demonstration showing predictions for different passenger profiles:

```bash
python example_prediction.py
```

**Output includes:**
- ✅ 1st Class Woman with Cabin → **SURVIVED** (95%+)
- ❌ 3rd Class Man without Cabin → **DID NOT SURVIVE** (low survival chance)
- ✅ Child with Family → **SURVIVED** (high chance - "women and children first")
- ✅ Young Woman (3rd Class) → **SURVIVED** (female advantage)

### Understanding the Features

The model expects engineered features from Phase 3. Here's the mapping:

| Feature | Description | Values |
|---------|-------------|--------|
| `Pclass` | Passenger class | 1-3 (1st/2nd/3rd) |
| `Sex` | Gender | 0 (Female), 1 (Male) |
| `HasCabin` | Has cabin number | 0 or 1 |
| `FamilySize` | Count of family members | 1-11 |
| `IsAlone` | Traveling solo | 0 or 1 |
| `Title_*` | One-hot encoded titles | Master, Miss, Mr, Mrs, Rare |
| `AgeGroup_*` | One-hot encoded age group | Child, Teen, Adult, Senior |
| `FareGroup_*` | One-hot encoded fare level | Low, Medium, High |
| `Embarked_*` | One-hot encoded port | C (Cherbourg), Q (Queenstown), S (Southampton) |

### Batch Predictions

To predict for multiple passengers:

```python
from src.predict import load_model, predict_batch
import pandas as pd

model = load_model("results/titanic_best_model.pkl")

# Load multiple passengers
passengers_df = pd.read_csv('data/train_engineered.csv')
# ... drop 'Survived' column if present

# Get predictions for all
results = predict_batch(model, passengers_df)
print(results[['Survived', 'Survival_Probability']])
```

##  Expected Results

- **Best Model:** XGBoost (typically 80-82% accuracy)
- **Training Time:** ~2-5 seconds
- **Prediction Speed:** ~100 predictions/second
- **Output:** Trained model saved as pickle file

##  Key Insights

- **Feature Importance:** Passenger class, age, and gender are strongest predictors
- **Class Balance:** ~38% survived (imbalanced dataset)
- **Missing Data:** Age has significant missingness (20%)
- **Model Performance:** Tree-based models outperform linear models on this dataset

##  Contributing

Contributions welcome! Feel free to:
- Add new models
- Improve feature engineering
- Optimize hyperparameters
- Add more visualizations
- Create additional analysis

##  License

This project is open source and available under the MIT License.

## 👤 Author

**Mohamed Amine Kallel**
- GitHub: [@MohammedAmine-dev](https://github.com/MohammedAmine-dev)
- Email: mohamkallel@gmail.com






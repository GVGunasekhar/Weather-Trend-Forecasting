# Weather Trend Forecasting

## Project Overview
This project analyzes the "Global Weather Repository" dataset to forecast future weather trends and showcase data science skills using both basic and advanced techniques. The dataset contains daily weather information for cities around the world with over 40 features reflecting global weather conditions.

## Objective
The main objective is to analyze weather data to forecast future trends while demonstrating proficiency in data science techniques. The project follows a structured approach with both basic and advanced assessment components.

## Dataset
The dataset used is the "Global Weather Repository" available on Kaggle: [World Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code)

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd weather-trend-forecasting
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, install packages individually:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow scipy
   pip install fbprophet  # For Prophet model
   ```

4. **Download the dataset:**
   - Download the "Global Weather Repository.csv" from the Kaggle link above
   - Place it in the `data/` directory of the project

5. **Verify installation:**
   ```bash
   python -c "import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; import seaborn as sns; import tensorflow as tf; from statsmodels.tsa.arima.model import ARIMA; print('All packages successfully imported!')"
   ```

## Methodology

### Basic Assessment

#### 1. Data Cleaning & Preprocessing
- Handled missing values using appropriate imputation techniques
- Identified and removed outliers using the IQR method
- Normalized numerical features to ensure consistent scales
- Converted timestamps to datetime format for time series analysis

#### 2. Exploratory Data Analysis (EDA)
- Generated correlation matrices to identify relationships between weather variables
- Created visualizations for temperature trends across different locations
- Analyzed temperature patterns over time using line plots
- Examined precipitation distribution with histograms and density plots
- Produced box plots to show temperature distribution by country

#### 3. Model Building
- Used ARIMA (AutoRegressive Integrated Moving Average) for time series forecasting
- Leveraged the `last_updated` feature as the time index for analysis
- Split data into training (80%) and testing (20%) sets
- Evaluated model performance using metrics:
  - Root Mean Square Error (RMSE)
  - Mean Absolute Error (MAE)
- Visualized actual vs. predicted values to assess forecast accuracy

### Advanced Assessment

#### 1. Advanced EDA - Anomaly Detection
- Implemented multiple outlier detection methods:
  - Z-Score method (identifying values with |Z| > 3)
  - Interquartile Range (IQR) method
  - Isolation Forest algorithm (unsupervised learning approach)
- Compared results across different detection methods
- Visualized outliers in time series data

#### 2. Forecasting with Multiple Models
- Developed and compared multiple forecasting models:
  - ARIMA (traditional time series approach)
  - Prophet (Facebook's forecasting tool)
  - LSTM (Long Short-Term Memory neural network)
- Created an ensemble model by averaging predictions from all models
- Evaluated each model using standard metrics (RMSE, MAE)
- Compared performance across individual models and the ensemble approach

#### 3. Unique Analyses
- **Climate Analysis**: Explored long-term temperature trends by resampling data to monthly intervals
- **Environmental Impact**: Analyzed correlations between weather parameters and air quality
- **Feature Importance**: Used Random Forest and permutation importance to identify key predictors
- **Spatial Analysis**: Visualized geographical temperature patterns using scatter plots
- **Geographical Patterns**: Compared temperature variations across different countries and regions

## Usage

### Running the Notebooks
```bash
google colab notebook notebooks/
```







```

## Key Findings
- Temperature patterns show significant variation across different geographical regions
- The ensemble forecasting approach delivered improved accuracy compared to individual models
- Specific anomalous weather events were identified through outlier detection
- Strong correlations were found between certain weather parameters
- Geographic location plays a crucial role in determining weather patterns

## Dependencies
```
pandas==1.3.5
numpy==1.21.6
matplotlib==3.5.3
seaborn==0.12.2
statsmodels==0.13.5
scikit-learn==1.0.2
tensorflow==2.11.0
fbprophet==0.7.1
scipy==1.7.3
jupyter==1.0.0
```


## Troubleshooting

### Common Issues

1. **Prophet Installation Issues**
   - On Windows, you might need Microsoft C++ Build Tools
   - Solution: Install from [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

2. **Memory Errors with Large Dataset**
   - Solution: Use data sampling or chunking techniques
   ```python
   # Sample approach
   df_sample = df.sample(frac=0.3, random_state=42)
   ```

3. **LSTM Model Errors**
   - Ensure TensorFlow is properly installed
   - Check input dimensions match expected format

## Results and Conclusions
The analysis successfully identified weather patterns and created forecasting models with good predictive power. The ensemble approach combining multiple forecasting techniques demonstrated superior performance compared to any single model. Geographic analysis revealed significant variations in weather patterns across different regions, highlighting the importance of location-specific forecasting approaches.

## Future Work
- Incorporate additional external data sources such as satellite imagery
- Implement more sophisticated deep learning architectures
- Develop interactive visualizations for exploring geographical patterns
- Analyze extreme weather events and their potential relationship to climate change
- Extend the forecasting horizon to provide longer-term predictions

## Contact
For questions or feedback about this project, please contact sekharg601@gmail.com .

# 🏆 IAI 2023 - 1st Place Solution: Forecasting Drug Demand in Hospital Supply Chains

[![Competition](https://img.shields.io/badge/Competition-IAI%202023-blue)](https://iai-competitions.ir/)
[![Rank](https://img.shields.io/badge/Rank-1st%20Place-gold)](https://github.com)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM%2FBiLSTM-green)](https://www.tensorflow.org/)
[![Publication](https://img.shields.io/badge/Published-JMSS%202025-success)](https://journals.lww.com/jmss)

This repository contains the **first-place solution** for the IAI 2023 competition on forecasting drug demand in hospital supply chains. The goal was to predict daily drug consumption across 12 hospitals in Hamedan province, Iran, for the period from January 1, 2023, to October 30, 2023.

**📄 Published Paper**: This work has been published in the *Journal of Medical Signals & Sensors* (2025). See [Citation](#citation) section below.

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Publication](#publication)
- [Citation](#citation)
- [Contributors](#contributors)

## 🎯 Problem Statement

The challenge was to forecast drug consumption for:
- **12 individual hospitals** in Hamedan province
- **Time period**: January 1, 2023 - October 30, 2023 (303 days)
- **Outputs required**:
  - Daily consumption for each hospital separately
  - Daily total consumption across all hospitals
  - Monthly total consumption across all hospitals

## 💡 Solution Overview

Our winning solution leverages a sophisticated ensemble of deep learning models with the following key innovations:

1. **Multi-Model Ensemble**: Combination of 4 different models per hospital with custom weights
2. **Advanced Feature Engineering**: Persian calendar features, weather data, COVID-19 impact, seasonal patterns
3. **Recursive Prediction Strategy**: Handles missing data by recursively predicting intermediate values
4. **Hospital-Specific Optimization**: Each hospital has its own trained models with custom ensemble weights

## 🏗️ Architecture

### Model Types

#### **Model 1** (LSTM-based with Dual Sequences)
- **Inputs**:
  - Sequence 1: Last 30 days from current year (15 features)
  - Sequence 2: Last 30 days from previous year (15 features)
  - Last year's consumption on same date
  - Last month's consumption on same date
  - Last week's consumption on same date
- **Architecture**:
  - BiLSTM layers for temporal sequences
  - Dense layers for scalar features
  - Feature concatenation and fusion
  - Output: Single value (predicted consumption)
- **Variants**: Best checkpoint + Last epoch checkpoint

#### **Model 4** (BiLSTM without Drug Feature)
- **Inputs**:
  - Sequence: Last 28 days (14 features, excluding drug consumption)
- **Architecture**:
  - BiLSTM layers for sequence processing
  - Dense layers for prediction
  - Output: Single value (predicted consumption)
- **Variants**: Best checkpoint + Last epoch checkpoint

### Ensemble Strategy

Each hospital uses a weighted ensemble of 4 models:
```
Final_Prediction = (Model_1_best × W1) + (Model_1_last × W2) + (Model_4_best × W3) + (Model_4_last × W4)
```

Custom weights per hospital (example for Hospital 1):
- Model_1 (best): 0.45
- Model_1 (last epoch): 0.30
- Model_4 (best): 0.10
- Model_4 (last epoch): 0.15

## 🔧 Features

### Temporal Features
- **Persian Calendar Features**:
  - Day of week (adjusted for Persian calendar)
  - Day of year (Persian calendar)
  - Day of month (Persian calendar)
- **School Holidays**: Binary indicator for school vacation periods
- **Weekly Holidays**: Weekend detection

### Environmental Features
- **Weather Data** (from Weatherbit API):
  - Maximum temperature
  - Minimum temperature
  - Average temperature
  - Relative humidity
  - Precipitation
  - Wind speed
  - Solar radiation
  - Snow
- **Seasonal Indicator**: 5-level categorical feature based on Persian seasons

### COVID-19 Impact
- **COVID Status**: 3-level indicator
  - 0: Normal period
  - 1: COVID-19 period (general impact)
  - 2: COVID-19 peak periods (high restrictions)

### Historical Consumption
- Daily consumption from previous years
- Temporal patterns (weekly, monthly, yearly)

## 📊 Dataset

### Training Data
- **Period**: January 1, 2019 - December 31, 2022 (4 years)
- **Records**: 14,290 rows
- **Hospitals**: 12 hospitals + 1 aggregate hospital (total 13)
- **Features**:
  - Hospital name (نام بیمارستان)
  - Day of week (روز هفته)
  - Consumption date (تاریخ مصرف)
  - Consumption count (تعداد مصرف)
  - Additional metadata (removed during preprocessing)

### External Data
- **Weather Data**: Historical weather from 2019-2023 (Hamedan region)
- **Model Weights**: Custom ensemble weights for each hospital

### Preprocessing Steps
1. **Outlier Removal**: Hospital-specific thresholds for abnormal consumption
2. **Duplicate Handling**: Summing duplicate records for same date
3. **Missing Date Filling**: Creating complete date range for all hospitals
4. **Feature Scaling**: MinMaxScaler for all numerical features
5. **Date Alignment**: Converting between Gregorian and Persian calendars

## 🤖 Models

### Training Configuration
- **Sequence Length**: 28-30 days
- **Framework**: TensorFlow/Keras
- **Layers**: LSTM, BiLSTM, Dense, Dropout
- **Activation**: ReLU (hidden), Linear (output)
- **Training Strategy**:
  - Separate models for each of 13 hospitals
  - Multiple checkpoints saved (best + last epoch)
  - Total: 52 models (13 hospitals × 4 model variants)

### Model Storage
Models are stored separately and loaded during inference:
```
Path_To_Models/
├── Model 1/
│   ├── M1_seq30_Hospital1.h5
│   ├── M1_seq30_Hospital1_LastEpoch.h5
│   └── ...
└── Model 4/
    ├── M4_seq28_Hospital1.h5
    ├── M4_seq28_Hospital1_LastEpoch.h5
    └── ...
```

## 📈 Results

### Competition Performance
- **Rank**: 🥇 **1st Place**
- **Prediction Period**: January 1, 2023 - October 30, 2023

### Output Files

#### 1. Daily Predictions - Separated Hospitals
- **File**: `Final Outputs/Daily-Separated-Hospitals.csv`
- **Format**: Hospital ID, Date, Predicted Consumption
- **Records**: 3,636 rows (12 hospitals × 303 days)

#### 2. Daily Predictions - All Hospitals Combined
- **File**: `Final Outputs/Daily-All-Hospitals.csv`
- **Format**: Date, Sum of Hospitals, Hospital 13 Prediction, Average
- **Records**: 303 rows
- **Method**: Average of (sum of 12 hospitals) and (hospital 13 prediction)

#### 3. Monthly Predictions - All Hospitals
- **File**: `Final Outputs/Monthly-All-Hospitals.csv`
- **Format**: Month, Total Consumption
- **Records**: 10 months (January - October 2023)
- **Example Output**:
  ```
  2023-01: 7,081
  2023-02: 6,042
  2023-03: 6,924
  ...
  2023-10: 6,421
  ```

## 🚀 Installation

### Requirements
```bash
pip install pandas numpy scikit-learn tensorflow jdatetime matplotlib seaborn requests
```

### Dependencies
- **Python**: 3.7+
- **TensorFlow**: 2.x
- **Pandas**: 1.3+
- **NumPy**: 1.20+
- **scikit-learn**: 0.24+
- **jdatetime**: Persian calendar support
- **requests**: Weather API calls

## 📖 Usage

### Google Colab (Recommended)

The solution was designed to run on Google Colab:

#### Training Notebook
- **Link**: [Hamedan - Train](https://colab.research.google.com/drive/1tctYTAZQv-AmvyPr-kX4XbBFxHGveMBC?usp=sharing)
- **Purpose**: Model training and development
- **Notebook**: `Hamedan_Final.ipynb`

#### Testing Notebook
- **Link**: [Hamedan - Test](https://colab.research.google.com/drive/1s9_bQFKrsFqm_Haj9JQVwxFibPC72ncn?usp=sharing)
- **Purpose**: Generate predictions for test period
- **Notebook**: `Hamedan_Test.ipynb`

### Running Locally

1. **Upload Required Files to Colab**:
   ```
   Upload these files to colab before running codes/
   ├── Train_drug demand_completed.csv
   ├── WeatherFrom2019To2023.csv
   ├── Model_weights.csv
   └── goly_drug_test_output.csv
   ```

2. **Set Weather API Key**:
   ```python
   WEATHER_API_KEY = "your_weatherbit_api_key"
   ```

3. **Run Training** (optional):
   ```python
   # Execute Hamedan_Final.ipynb
   # This will train all models for all hospitals
   ```

4. **Run Prediction**:
   ```python
   # Execute Hamedan_Test.ipynb
   # This will generate predictions and save to CSV files
   ```

### Key Functions

#### Data Generation
```python
# Generate training dataset for a date range
merged_input_1_np, merged_input_2_np, merged_LY_L, merged_LM_L, merged_LW_L = generateDataset(
    firstLabelDate,
    lastLabelDate,
    hospitalName,
    seqSize=30
)
```

#### Prediction
```python
# Make weighted ensemble prediction
prediction = Weighted_predict(
    model_list,      # List of 4 models for the hospital
    labelDate,       # Date to predict
    HospitalName,    # Hospital ID (1-13)
    debug_showPrints=False
)
```

#### Recursive Prediction
The system automatically handles missing intermediate predictions by recursively predicting required dates.

## 📁 File Structure

```
IAI-2023-1st-place-Forecasting-Drug-Demand-in-Hospital-Supply-Chains/
│
├── README.md                                      # This file
├── Colab Codes.txt                               # Links to Colab notebooks
├── report (2).docx                               # Competition report
│
├── Hamedan_Final.ipynb                           # Training notebook
├── Hamedan_Test.ipynb                            # Testing/prediction notebook
│
├── Upload these files to colab before running codes/
│   ├── Train_drug demand_completed.csv           # Training data (2019-2022)
│   ├── WeatherFrom2019To2023.csv                # Weather data
│   ├── Model_weights.csv                         # Ensemble weights per hospital
│   └── goly_drug_test_output.csv                # Test dataset template
│
└── Final Outputs/
    ├── Daily-All-Hospitals.csv                   # Daily predictions (combined)
    ├── Daily-Separated-Hospitals.csv             # Daily predictions (per hospital)
    ├── Monthly-All-Hospitals.csv                 # Monthly predictions
    └── drug_test_output.xlsx                     # Final submission file
```

## 🔬 Methodology Highlights

### 1. Data Preprocessing
- Outlier detection and removal (hospital-specific thresholds)
- Duplicate record aggregation
- Complete date range generation (no missing dates)
- Persian calendar integration
- MinMax feature scaling

### 2. Feature Engineering
- **Temporal**: Persian day/month/year, weekday
- **Environmental**: 8 weather parameters from Weatherbit API
- **Categorical**: School holidays, COVID-19 status, seasonal indicators
- **Historical**: Last year/month/week consumption patterns

### 3. Model Architecture
- **LSTM layers**: Capture long-term temporal dependencies
- **BiLSTM layers**: Capture bidirectional patterns
- **Dense layers**: Feature fusion and non-linear transformations
- **Dropout layers**: Regularization
- **Multi-input design**: Combine sequences and scalar features

### 4. Ensemble Strategy
- 4 models per hospital (2 architectures × 2 checkpoints)
- Custom weights optimized per hospital
- Weighted average for final prediction

### 5. Recursive Prediction
- Handles missing historical data dynamically
- Predicts intermediate values when needed
- Ensures complete feature availability

### 6. Validation Strategy
- Hospital 13 (aggregate) used as validation
- Cross-comparison with sum of individual hospitals
- Final prediction: average of both approaches

## 🎓 Key Innovations

1. **Persian Calendar Integration**: Proper handling of Persian holidays, weekends, and seasonal patterns
2. **COVID-19 Modeling**: Multi-level encoding of pandemic impact periods
3. **Weather Integration**: Real-time API integration for missing weather data
4. **Hospital-Specific Optimization**: Custom ensemble weights per hospital
5. **Recursive Prediction**: Smart handling of missing intermediate predictions
6. **Multi-Checkpoint Ensemble**: Using both best and last epoch models

## 🏅 Competition Details

- **Competition**: IAI 2023 (Isfahan AI Competition / Iranian AI Competition)
- **Challenge**: Forecasting Drug Demand in Hospital Supply Chains
- **Region**: Hamedan Province, Iran
- **Hospitals**: 12 hospitals
- **Timeframe**: 10-month prediction (Jan-Oct 2023)
- **Result**: 🥇 **1st Place**

## 📄 Publication

This work has been published in the peer-reviewed journal:

**Title**: Isfahan Artificial Intelligence Event 2023: Drug Demand Forecasting

**Journal**: Journal of Medical Signals & Sensors
**Volume**: 15, Issue 1
**Pages**: 10.4103
**Publisher**: Medknow
**Publication Date**: January 2025

**Abstract**:
> **Background**: The pharmaceutical industry has seen increased drug production by different manufacturers. Failure to recognize future needs has caused improper production and distribution of drugs throughout the supply chain of this industry. Forecasting demand is one of the basic requirements to overcome these challenges. Forecasting the demand helps the drug to be well estimated and produced at a certain time.
>
> **Methods**: Artificial intelligence (AI) technologies are suitable methods for forecasting demand. The more accurate this forecast is, the better it will be to decide on the management of drug production and distribution. Isfahan AI competitions-2023 have organized a challenge to provide models for accurately predicting drug demand. In this article, we introduce this challenge and describe the proposed approaches that led to the most successful results.
>
> **Results**: A dataset of drug sales was collected in 12 hospitals in Hamedan province, covering multiple years of historical data. This challenge attracted various AI-based solutions, with the top-performing approaches utilizing deep learning architectures including LSTM and BiLSTM networks combined with advanced feature engineering.

## 📝 Citation

If you use this work, please cite the published paper:

```bibtex
@article{jahani2025isfahan,
  title={Isfahan Artificial Intelligence Event 2023: Drug Demand Forecasting},
  author={Jahani, Meysam and Zojaji, Zahra and Montazerolghaem, AhmadReza and Palhang, Maziar and Ramezani, Reza and Golkarnoor, Ahmadreza and Akhavan Safaei, Alireza and Bahak, Hossein and Saboori, Pegah and Soufi Halaj, Behnam and Naghsh-Nilchi, Ahmad R and Mohamadpoor, Fatemeh and Jafarizadeh, Saeid},
  journal={Journal of Medical Signals \& Sensors},
  volume={15},
  number={1},
  pages={10.4103},
  year={2025},
  publisher={Medknow}
}
```

**Note**: For citing this specific implementation repository:

```bibtex
@misc{iai2023-implementation,
  title={IAI 2023 1st Place Solution: Implementation of Drug Demand Forecasting},
  author={Akhavan Safaei, Alireza and team},
  year={2023},
  publisher={GitHub},
  howpublished={\url{https://github.com/akhavansafaei/IAI-2023-1st-place-Forecasting-Drug-Demand-in-Hospital-Supply-Chains}}
}
```

## 🤝 Contributors

This solution was developed by a collaborative team for the IAI 2023 competition:

**Team Members** (as listed in the published paper):
- Meysam Jahani
- Zahra Zojaji
- AhmadReza Montazerolghaem
- Maziar Palhang
- Reza Ramezani
- Ahmadreza Golkarnoor
- **Alireza Akhavan Safaei**
- Hossein Bahak
- Pegah Saboori
- Behnam Soufi Halaj
- Ahmad R Naghsh-Nilchi
- Fatemeh Mohamadpoor
- Saeid Jafarizadeh

This collaborative effort achieved **1st place** in the IAI 2023 drug demand forecasting challenge.

## 📧 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Note**: This solution requires access to Weatherbit API for weather data. You can obtain a free API key at [https://www.weatherbit.io/](https://www.weatherbit.io/).

## 🔐 License

This project is part of the IAI 2023 competition submission. Please check competition rules for usage restrictions.

---

**Built with**: TensorFlow, Keras, Python, Persian Calendar, Weather APIs

**Competition**: IAI 2023 | **Status**: 🥇 1st Place Winner

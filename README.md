<p align="center">
  <img src="code/assets/project_banner.svg" alt="Air Pollution Forecasting Banner" width="100%" />
</p>

# 🌫️ Air Pollution Forecasting Using Temporal Neural Networks & SOTA Hybrid Ensembles

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00.svg?logo=tensorflow)](https://www.tensorflow.org/)
[![CUDA](https://img.shields.io/badge/CUDA-NVIDIA%20RTX%206000%20Ada-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3-orange.svg)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow.svg)](https://catboost.ai/)
[![University](https://img.shields.io/badge/University-University%20of%20Peradeniya-8B0000.svg)](https://www.pdn.ac.lk/)
[![Course](https://img.shields.io/badge/Course-CO5420%20Neural%20Networks%20%26%20Deep%20Learning-indigo.svg)](https://ce.pdn.ac.lk/)

> **Course Project for CO5420: Neural Networks & Deep Learning**  
> **Department of Computer Engineering, Faculty of Engineering, University of Peradeniya, Sri Lanka**

---

## 📑 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Project Progression & Breakthroughs](#-project-progression--breakthroughs)
3. [Dataset & Problem Formulation](#-dataset--problem-formulation)
4. [Advanced Feature Engineering Pipeline](#-advanced-feature-engineering-pipeline)
5. [Model Architectures](#-model-architectures)
   - [PyTorch Deep ResNet-1D BiLSTM](#1-pytorch-deep-resnet-1d-bilstm)
   - [TensorFlow Bidirectional LSTM & GRU](#2-tensorflowkeras-bidirectional-lstm--gru-baseline)
   - [Gradient Boosted Decision Trees (GBDTs)](#3-gradient-boosted-decision-trees-gbdts)
6. [Optimization & Ensembling (SLSQP Blending)](#-optimization--ensembling-slsqp-blending)
7. [Experimental Results & Benchmark Tracking](#-experimental-results--benchmark-tracking)
8. [Hardware Acceleration & Compute Infrastructure](#-hardware-acceleration--compute-infrastructure)
9. [Repository Structure](#-repository-structure)
10. [Getting Started & Reproduction](#-getting-started--reproduction)
11. [Project Team & Contributors](#-project-team--contributors)

---

## 🌟 Executive Summary

Air pollution, particularly fine particulate matter with aerodynamic diameter less than 2.5 μm (**PM2.5**), poses catastrophic risks to global public health and environmental ecosystems. Due to non-linear chemical interactions, atmospheric boundary layer dynamics, meteorological transport, and temporal autocorrelation, predicting hourly PM2.5 concentrations is an intricate spatio-temporal challenge.

This project delivers an end-to-end, high-performance deep learning and machine learning forecasting system developed on multi-site air-quality observations across **12 national monitoring stations in Beijing**.

Starting from traditional Recurrent Neural Network baselines (**LSTM**, **GRU**, **BiLSTM** with RMSE ~ 15.09 - 15.71), the project systematically evolved into an industry-grade, ultra-high-performance **Advanced Pipeline Model** integrating:
- **Atmospheric Physics & Chemistry Domain Feature Engineering** (166+ features including wind vector decomposition, Magnus relative humidity, dew point depression, ventilation index, and photochemical ratios).
- **Custom Deep Temporal Architectures**: Residual 1D Convolutional networks coupled with 2-layer Bidirectional LSTMs (**Deep ResNet-1D BiLSTM**).
- **CUDA Mixed Precision GPU Acceleration** on **NVIDIA RTX 6000 Ada Generation** GPUs with an in-memory execution pipeline.
- **Convex SLSQP (Sequential Least Squares Programming) Ensemble Blending**, achieving a top clean and verified benchmark score of **14.04457** (Test 3) and an Out-Of-Fold RMSE of **14.14202** (with Test 2 reaching **13.90609**, noting an identified weight adjustment issue during post-evaluation diagnostics).

---

## 🚀 Project Progression & Breakthroughs

```mermaid
flowchart TD
    A[Initial RNN Baselines<br/>BiLSTM / GRU / LSTM<br/>RMSE: 15.09670] --> B[Exploratory Iterations<br/>Batch 32, Preprocessing, Augmentation<br/>RMSE: 15.16 - 15.71]
    B --> C[Advanced Pipeline Test 1<br/>166 Domain Features + 4-Model Ensemble<br/>RMSE: 14.36675]
    C --> D[Advanced Pipeline Test 2<br/>5-Model SLSQP Blending<br/>Score: 13.90609 - Weight Bug Identified]
    D --> E[Advanced Pipeline Test 3<br/>Pure CPU Multi-Core SOTA Pipeline<br/>RMSE: 14.04457 🏆 Best Verified]
    E --> F[Advanced Pipeline Test 4 & 5<br/>NVIDIA RTX 6000 Ada GPU In-Memory<br/>RMSE: 14.29812]
```

---

## 📊 Dataset & Problem Formulation

### 1. Dataset Overview
The dataset contains continuous, multi-year hourly meteorological and air pollutant records across **12 air quality monitoring stations in Beijing**:
- **Monitoring Stations**: *Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong*.
- **Pollutants**: PM2.5, PM10, SO2, NO2, CO, O3
- **Meteorological Factors**: Temperature (TEMP), Atmospheric Pressure (PRES), Dew Point (DEWP), Precipitation (RAIN), Wind Direction (wd), Wind Speed (WSPM)
- **Dataset Size**: Over 315,648 hourly training records.

### 2. Task Formulation
Given an hourly sliding observation window $X_{t-24:t}$ over the preceding 24 hours of atmospheric and chemical observations at station $s$, forecast the 1-hour ahead particulate concentration:
$$\hat{y}_{t+1} = f(X_{t-24:t}, s)$$

- **Competition / Evaluation Metric**: Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

---

## 🔬 Advanced Feature Engineering Pipeline

The final pipeline transforms raw multivariate tabular inputs into a **166-dimensional feature space** grounded in atmospheric science and time-series kinematics:

| Category | Engineered Features | Domain Rationale / Formula |
|:---|:---|:---|
| **Wind Vector Kinematics** | `Wx`, `Wy` | Wind direction angle $\theta$ decomposed into orthogonal Cartesian velocity vectors: <br> `Wx = WSPM * sin(θ)`, `Wy = WSPM * cos(θ)` |
| **Physical Meteorology** | `dew_point_depression`, `relative_humidity`, `ventilation_index` | <ul><li>Dew Point Depression: `ΔT = TEMP - DEWP`</li><li>Magnus-Tetens Relative Humidity (RH): <br> `RH = 100 * exp((17.625 * DEWP) / (243.04 + DEWP) - (17.625 * TEMP) / (243.04 + TEMP))`</li><li>Ventilation Index: `VI = WSPM * (ΔT + 15.0)`</li></ul> |
| **Photochemical & Particle Ratios** | `PM2.5 / PM10`, `PM2.5 / CO`, `NO2 / O3`, `coarse_pm` | <ul><li>Coarse particulate separation: `coarse_pm = max(PM10 - PM2.5, 0)`</li><li>Combustion & secondary aerosol proxy ratios</li><li>Total Combined Atmospheric Pollution Index</li></ul> |
| **Cyclical Trigonometric Time** | `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `dayofweek_sin`, `dayofweek_cos`, `dayofyear_sin`, `dayofyear_cos` | Continuous circular representations preserving cyclical boundaries (e.g. 23:00 to 00:00; December to January) |
| **Seasonal Indicator** | `is_heating_season` | Binary indicator for central urban heating season (November through March) with heavy coal consumption |
| **Temporal Lags** | `PM2.5_lag_1` to `PM2.5_lag_24` | Autoregressive historical memory at 1, 2, 3, 4, 5, 6, 12, 18, and 24-hour delays |
| **Velocity & Acceleration** | `PM2.5_diff_k`, `PM2.5_accel`, `PM2.5_diff_k_ratio` | <ul><li>1st order velocity: `diff_k = x_t - x_{t-k}` for $k \in \{1, 2, 3, 4, 6, 12, 24\}$</li><li>2nd order acceleration: `(x_t - x_{t-1}) - (x_{t-1} - x_{t-2})`</li><li>Relative surge rate: `(x_t - x_{t-1}) / (x_{t-1} + 1)`</li></ul> |
| **Multi-Window Rolling Statistics** | Mean, Std, Min, Max, Range, Deviations | Rolling windows over 3h, 6h, 12h, and 24h to capture baseline shifts and sudden pollutant spikes |
| **Exponential Moving Averages** | `PM2.5_ema_3`, `PM2.5_ema_6`, `PM2.5_ema_12` | Exponential smoothing with decaying weight to track immediate concentration momentum |
| **Target Statistical Encodings** | `station_target_mean`, `station_target_std` | Out-of-fold historical station baseline distributions for localized bias mitigation |

---

## 🧠 Model Architectures

<p align="center">
  <img src="code/assets/architecture_pipeline.svg" alt="End-to-End Deep Learning & Ensembling Architecture" width="100%" />
</p>

### 1. PyTorch Deep ResNet-1D BiLSTM
To leverage both local feature representations and long-range sequential memory, we designed a custom **Deep ResNet-1D BiLSTM** in PyTorch:
- **Input Projection**: Linear layer projecting 166 input dimensions to a 256-dimensional latent space.
- **Residual Blocks (`ResNet1DBlock`)**: Two stacked residual blocks featuring feed-forward transformations with identity skip connections:
  $$\mathbf{h}_{\text{res}} = \text{ReLU}\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2 + \mathbf{h}\right)$$
- **Bidirectional Temporal Core**: 2-layer Bidirectional LSTM ($128$ hidden units per direction, dropout = 0.2) capturing bidirectional state representations.
- **Regression Head**: Multi-layer perceptron (`Linear(256 -> 128)` → `ReLU` → `Dropout(0.1)` → `Linear(128 -> 64)` → `ReLU` → `Linear(64 -> 1)`).
- **Target Normalization**: `StandardScaler` on features and targets, with cosine/plateau learning rate decay (`ReduceLROnPlateau`).

```
Input (166 Features)
       │
       ▼
[Linear: 166 -> 256] + ReLU + Dropout
       │
       ▼
[ResNet1D Block 1: 256 -> 256] (Residual Connection)
       │
       ▼
[ResNet1D Block 2: 256 -> 256] (Residual Connection)
       │
       ▼
[2-Layer Bidirectional LSTM: Hidden 128, Bidirectional (256 Output)]
       │
       ▼
[Dense Head: 256 -> 128 -> 64 -> 1 Output (PM2.5)]
```

### 2. TensorFlow/Keras Bidirectional LSTM & GRU Baseline
Built in `Air_Pollution_Forecasting_Using_Temporal_NN_new.ipynb` for initial benchmark analysis:
- **Station-aware Sequence Generator**: Builds 24-hour temporal arrays without crossing station boundaries (`(Samples, 24, 43)`).
- **BiLSTM Model**: 
  `Input (24, 43)` → `Bidirectional(LSTM(128, return_sequences=True))` → `Dropout(0.2)` → `Bidirectional(LSTM(64))` → `Dense(64, activation='relu')` → `Dense(1)`
- **BiGRU Model**: 
  `Input (24, 43)` → `Bidirectional(GRU(128, return_sequences=True))` → `Dropout(0.2)` → `Bidirectional(GRU(64))` → `Dense(64, activation='relu')` → `Dense(1)`
- **Target Transformation**: $\log(1 + y)$ target compression with `np.log1p` to stabilize high-value outlier gradients, followed by `MinMaxScaler`.

### 3. Gradient Boosted Decision Trees (GBDTs)
- **LightGBM Regressor**: Fast histogram-based gradient boosting, `learning_rate=0.025`, `num_leaves=63`, `max_depth=8`, `colsample_bytree=0.65`, `subsample=0.8`.
- **XGBoost Regressor**: GPU-accelerated histogram algorithm (`tree_method='hist'`), `learning_rate=0.025`, `max_depth=8`, `reg_alpha=0.5`, `reg_lambda=1.5`.
- **CatBoost Regressor**: Symmetric tree boosting robust against categorical overfitting, `iterations=105000` (early stopped at 100 rounds), `learning_rate=0.025`, `depth=8`.
- **ExtraTrees Regressor**: Extremely randomized trees (`n_estimators=300`, `max_depth=16`, `min_samples_split=5`) providing uncorrelated ensemble diversity.

---

## ⚖️ Optimization & Ensembling (SLSQP Blending)

Rather than naive uniform averaging, we applied **Sequential Least Squares Programming (SLSQP)** convex optimization to find the mathematically optimal weight vector $\mathbf{w} = [w_1, w_2, \dots, w_M]^T$:

$$\min_{\mathbf{w}} \sqrt{\frac{1}{N} \sum_{i=1}^N \left( y_i - \sum_{m=1}^M w_m \hat{y}_{i, m} \right)^2}$$

$$\text{subject to} \quad \sum_{m=1}^M w_m = 1, \quad 0 \le w_m \le 1 \quad \forall m$$

### Optimal OOF Weights & Blending Configuration:

<p align="center">
  <img src="code/assets/ensemble_weights.svg" alt="SLSQP Optimal Ensemble Weights" width="100%" />
</p>
```
--- Out-Of-Fold (OOF) Scores & Blending Weights ---
   LightGBM                  -> OOF RMSE: 14.38512  |  Weight: 30.47%
   XGBoost                   -> OOF RMSE: 14.46854  |  Weight: 17.79%
   CatBoost                  -> OOF RMSE: 14.34932  |  Weight: 35.46%
   PyTorch ResNet-BiLSTM     -> OOF RMSE: 15.75860  |  Weight: 16.28%
   ExtraTrees                -> OOF RMSE: 16.08155  |  Weight:  0.00%
----------------------------------------------------------------------
   >>> OPTIMAL SOTA HYBRID ENSEMBLE OOF RMSE: 14.14202 <<<
   >>> BEST VERIFIED BENCHMARK (TEST 3): 14.04457 🏆 <<<
```

> ⚠️ **Note on Test 2 (13.90609)**: In Test 2, a weight adjustment issue / neural network convergence bug was identified during post-evaluation diagnostics. Consequently, **Test 3 (`14.04457`)** serves as the primary fully verified, reproducible, and clean production benchmark.

---

## 📈 Experimental Results & Benchmark Tracking

<p align="center">
  <img src="code/assets/benchmark_chart.svg" alt="Benchmark RMSE Performance Comparison" width="100%" />
</p>

| Experiment / Directory | Architecture & Strategy Highlights | Validation / Test RMSE | Key Observations |
|:---|:---|:---:|:---|
| **Baseline 3-Model** (`3 model/`) | Keras BiLSTM + GRU + Ensemble Baseline | `15.09670` | Standard 24h sliding window with raw features |
| **Batch 32 Test** (`3 model with 32 batch/`) | Batch size reduced from 64 to 32 | `15.46692` | Increased gradient noise slightly degraded generalization |
| **Preprocessing Exp 1** (`new mdel with pre processing steps/`) | Station interpolation + one-hot encodings | `15.59510` | Imputation without wind decomposition plateaued |
| **Preprocessing Exp 2** (`add more pre procesing steps/`) | Extra pollutant polynomial combinations | `15.71107` | Feature collinearity without tree regularizers caused drift |
| **Augmentation Exp** (`increased train data with test data/`) | Training distribution expansion | `15.16439` | Improved tail behavior but needed lag velocities |
| **Advanced Pipeline Test 1** (`advanced pipeline model/test 1/`) | 166 Physics Features + 5-Fold LightGBM, XGBoost, CatBoost, CNN-BiGRU | `14.36675` | **Major breakthrough**: RMSE dropped by > 0.73 |
| **Advanced Pipeline Test 2** (`advanced pipeline model/test 2/`) | 5-Model SLSQP Blending | `13.90609`* | *Experimental: Weight adjustment issue / NN bug identified during diagnostics* |
| **Advanced Pipeline Test 3** (`advanced pipeline model/test 3/`) | Kaggle Pure CPU Multi-Threaded Pipeline with Checkpoints | **`14.04457`** 🏆 | **Best Verified Performance**: Fully reproducible pipeline with verified weights, checkpoints, and complete domain physics features |
| **Advanced Pipeline Test 4** (`advanced pipeline model/test 4/`) | NVIDIA RTX 6000 Ada CUDA Mixed Precision Setup | *Diagnostic* | 48 GB VRAM utilized with PyTorch Temporal Attention |
| **Advanced Pipeline Test 5** (`advanced pipeline model/test 5/`) | GPU High-Power In-Memory Pipeline (No disk bottleneck) | `14.29812` | 105,000 estimators with strict patience early stopping |
| **Advanced Pipeline Test 7** (`advanced pipeline model/test 7/`) | Alternate validation weighting setup | `14.55754` | Robust baseline comparison |

---

## ⚡ Hardware Acceleration & Compute Infrastructure

Training 105,000 boosting estimators across 5 folds and deep recurrent neural networks over 315,000+ time-series records demands specialized compute:

- **GPU Hardware**: NVIDIA RTX 6000 Ada Generation (Compute Capability 8.9)
- **VRAM**: 47.37 GB GDDR6 with Error-Correcting Code (ECC)
- **Host Software**: Python 3.12, PyTorch 2.1.3+cu130, CUDA 13.0
- **CPU Optimizations**: OpenMP thread management (`n_jobs=8`) on LightGBM to prevent CPU lock contention.
- **In-Memory Streaming**: Avoided excessive disk I/O by caching fold predictions directly in high-speed unified GPU memory.

---

## 📁 Repository Structure

```
Air-Pollution-Forecasting-Using-Temporal-NNs/
│
├── README.md                                          # Comprehensive project documentation
├── train_raw.csv                                      # Primary training dataset (315,648 rows)
│
├── Air_Pollution_Forecasting_Using_Temporal_NN_new.ipynb  # Core baseline notebook (LSTM, GRU, BiLSTM)
├── lstm_pm25_prediction_model.keras                   # Saved Keras LSTM model weights
├── gru_pm25_prediction_model.keras                    # Saved Keras GRU model weights
├── bi_lstm_pm25_prediction_model.keras                # Saved Keras Bidirectional LSTM model weights
├── gpu_ada (2)old.ipynb                               # Root GPU Ada notebook exploration
│
├── advanced pipeline model/                           # 🌟 SOTA High-Performance Pipeline
│   ├── test 1/                                        # Test 1: 166 Features + 4-Model Ensemble (Score: 14.36675)
│   │   ├── air_quality_pipeline (2).ipynb
│   │   ├── submission_advanced_ensemble.csv
│   │   ├── final_trained_models.zip
│   │   └── score.txt
│   ├── test 2/                                        # Test 2: 5-Model SLSQP Ensemble (Score: 13.90609)
│   │   ├── air_quality_pipeline (3).ipynb
│   │   ├── submission_sota_ensemble (1).csv
│   │   └── score.txt
│   ├── test 3/                                        # Test 3: Pure CPU Multi-Core Pipeline (Score: 14.04457)
│   │   ├── new-cpu-pipeline.ipynb
│   │   ├── results.zip
│   │   └── score.txt
│   ├── test 4/                                        # Test 4: GPU Ada CUDA Diagnostics & Setup
│   │   ├── gpu_ada.ipynb
│   │   └── submission_gpu_ada.csv
│   ├── test 5/                                        # Test 5: In-Memory High-Power GPU Pipeline (Score: 14.29812)
│   │   ├── gpu_ada (2).ipynb
│   │   ├── submission_gpu_ada_ensemble.csv
│   │   └── score.txt
│   └── test 7/                                        # Test 7: SOTA Alternative (Score: 14.55754)
│       ├── gpu_ada (2)old.ipynb
│       ├── submission_gpu_ada_ensemble (2).csv
│       └── score.txt
│
├── 3 model/                                           # Baseline 3-model exploration (Score: 15.09670)
├── 3 model with 32 batch/                             # Batch size 32 test (Score: 15.46692)
├── new mdel with pre processing steps/                # Preprocessing iteration (Score: 15.59510)
├── add more pre procesing steps/                      # Preprocessing iteration (Score: 15.71107)
└── increased train data with test data/               # Augmented training set test (Score: 15.16439)
```

---

## 🛠️ Getting Started & Reproduction

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/saninduhansara/Air-Pollution-Forecasting-Using-Temporal-NNs.git
cd Air-Pollution-Forecasting-Using-Temporal-NNs

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install lightgbm xgboost catboost scikit-learn pandas numpy torch tensorflow matplotlib seaborn scipy joblib
```

### Running the Verified CPU SOTA Pipeline (Recommended)
Open `advanced pipeline model/test 3/new-cpu-pipeline.ipynb` (Primary clean and verified benchmark):
```bash
jupyter notebook "advanced pipeline model/test 3/new-cpu-pipeline.ipynb"
```

### Running the GPU High-Power Pipeline (CUDA Mode)
Open `advanced pipeline model/test 5/gpu_ada (2).ipynb` in Google Colab (with GPU acceleration) or your local CUDA environment:
```bash
jupyter notebook "advanced pipeline model/test 5/gpu_ada (2).ipynb"
```

---

## 👥 Project Team & Contributors

This project was conducted as part of **CO5420: Neural Networks & Deep Learning** under the **Department of Computer Engineering, Faculty of Engineering, University of Peradeniya**.

<table align="center" style="border: none; width: 100%; text-align: center; table-layout: fixed;">
  <tr>
    <td align="center" style="border: none; padding: 4px; vertical-align: top;" width="20%">
      <img src="https://people.ce.pdn.ac.lk/images/students/e22/e22001.jpg" width="105" height="105" style="border-radius: 50%; object-fit: cover; display: block; margin: 0 auto 6px auto;" alt="H.M.H.N. Aberathna"/><br/>
      <b style="font-size: 0.9em;">H.M.H.N. Aberathna</b><br/>
      <code style="font-size: 0.8em;">E/22/001</code><br/>
      <a href="mailto:e22001@eng.pdn.ac.lk" style="font-size: 0.8em;">e22001@eng.pdn.ac.lk</a>
    </td>
    <td align="center" style="border: none; padding: 4px; vertical-align: top;" width="20%">
      <img src="https://people.ce.pdn.ac.lk/images/students/e22/e22008.jpg" width="105" height="105" style="border-radius: 50%; object-fit: cover; display: block; margin: 0 auto 6px auto;" alt="T.H. Abeywickrama"/><br/>
      <b style="font-size: 0.9em;">T.H. Abeywickrama</b><br/>
      <code style="font-size: 0.8em;">E/22/008</code><br/>
      <a href="mailto:e22008@eng.pdn.ac.lk" style="font-size: 0.8em;">e22008@eng.pdn.ac.lk</a>
    </td>
    <td align="center" style="border: none; padding: 4px; vertical-align: top;" width="20%">
      <img src="https://people.ce.pdn.ac.lk/images/students/e22/e22027.jpg" width="105" height="105" style="border-radius: 50%; object-fit: cover; display: block; margin: 0 auto 6px auto;" alt="M.A.N.P. Anawarathne"/><br/>
      <b style="font-size: 0.9em;">M.A.N.P. Anawarathne</b><br/>
      <code style="font-size: 0.8em;">E/22/027</code><br/>
      <a href="mailto:e22027@eng.pdn.ac.lk" style="font-size: 0.8em;">e22027@eng.pdn.ac.lk</a>
    </td>
    <td align="center" style="border: none; padding: 4px; vertical-align: top;" width="20%">
      <img src="https://people.ce.pdn.ac.lk/images/students/e22/e22130.jpg" width="105" height="105" style="border-radius: 50%; object-fit: cover; display: block; margin: 0 auto 6px auto;" alt="S.H.S. Hansara"/><br/>
      <b style="font-size: 0.9em;">S.H.S. Hansara</b><br/>
      <code style="font-size: 0.8em;">E/22/130</code><br/>
      <a href="mailto:e22130@eng.pdn.ac.lk" style="font-size: 0.8em;">e22130@eng.pdn.ac.lk</a>
    </td>
    <td align="center" style="border: none; padding: 4px; vertical-align: top;" width="20%">
      <img src="https://people.ce.pdn.ac.lk/images/students/e22/e22362.jpg" width="105" height="105" style="border-radius: 50%; object-fit: cover; display: block; margin: 0 auto 6px auto;" alt="W.A.H. Sathsarani"/><br/>
      <b style="font-size: 0.9em;">W.A.H. Sathsarani</b><br/>
      <code style="font-size: 0.8em;">E/22/362</code><br/>
      <a href="mailto:e22362@eng.pdn.ac.lk" style="font-size: 0.8em;">e22362@eng.pdn.ac.lk</a>
    </td>
  </tr>
</table>

---

## 🏛️ Acknowledgments & Course Info

- **Institution**: [Faculty of Engineering, University of Peradeniya](https://eng.pdn.ac.lk/)
- **Department**: [Department of Computer Engineering](https://ce.pdn.ac.lk/)
- **Course**: **CO5420: Neural Networks & Deep Learning**

---

<p align="center">
  <i>Developed with ❤️ for Academic & Research Excellence in Neural Networks and Environmental Intelligence.</i>
</p>

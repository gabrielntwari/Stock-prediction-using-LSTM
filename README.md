# Apple Stock Market Prediction with LSTM

**Deep Learning Time Series Forecasting for Apple Stock Prices**

This project implements a Long Short-Term Memory (LSTM) neural network to predict Apple Inc. (AAPL) stock prices using historical market data. The model outperforms traditional moving average baselines through advanced sequential pattern recognition.

## 📈 Project Overview

This deep learning project focuses on predicting Apple stock prices using LSTM networks, a powerful recurrent neural network architecture specifically designed for time series data. The model analyzes historical price patterns to forecast future stock movements with superior accuracy compared to traditional statistical methods.

## 🎯 Objectives

- **Build an LSTM model** for Apple stock price prediction
- **Compare performance** against moving average baselines
- **Implement proper time series validation** with temporal data splitting
- **Evaluate model robustness** using statistical significance tests
- **Create a forecasting pipeline** for future price predictions

## 🚀 Key Features

### Data Pipeline
- **Historical Data Collection**: 24+ years of Apple stock data (2000-2024) via Yahoo Finance API
- **Time Series Windowing**: 5-day lookback windows for sequence learning
- **Proper Temporal Split**: 80/20 train-test split maintaining chronological order
- **Data Normalization**: MinMax scaling for optimal LSTM convergence

### Model Architecture
- **LSTM Layers**: Sequential memory cells for pattern recognition
- **Dropout Regularization**: Prevents overfitting with 20% dropout rate
- **Batch Normalization**: Stabilizes training process
- **Adam Optimizer**: Adaptive learning rate optimization
- **Custom Loss Function**: Mean Squared Error for regression

### Advanced Evaluation
- **Multiple Metrics**: MSE, RMSE, MAE for comprehensive assessment
- **Baseline Comparison**: 50-day Moving Average benchmark
- **Statistical Significance**: Diebold-Mariano test for model superiority
- **Forecasting Function**: Multi-step ahead prediction capability

## 📊 Model Performance

### LSTM vs Moving Average Results:

| Metric | LSTM Model | Moving Average (50-day) | Improvement |
|--------|------------|-------------------------|-------------|
| **MSE** | 45.14 | 76.54 | **41% Better** |
| **RMSE** | 6.72 | 8.75 | **23% Better** |
| **MAE** | 5.27 | 6.61 | **20% Better** |

### Statistical Significance
- **Diebold-Mariano Test**: Statistic = 14.49, p-value < 0.001
- **Result**: LSTM significantly outperforms moving average baseline

## 🛠 Technical Implementation

### Technologies Used
- **Python** - Core programming language
- **TensorFlow/Keras** - Deep learning framework
- **yfinance** - Financial data retrieval
- **scikit-learn** - Data preprocessing and metrics
- **NumPy/Pandas** - Data manipulation
- **Matplotlib** - Data visualization
- **statsmodels** - Statistical testing

### Model Architecture Details
```python
# LSTM Network Structure
Sequential([
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    BatchNormalization(),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)  # Output layer
])
```

### Data Preprocessing Pipeline
1. **Historical Data Download** - 24 years of AAPL data
2. **Feature Engineering** - OHLCV price features
3. **Sequence Creation** - 5-day input windows
4. **Normalization** - MinMax scaling [0,1]
5. **Train-Test Split** - Temporal preservation
6. **3D Reshaping** - LSTM input format preparation

## 📈 Key Insights

### Model Performance
- **Superior Accuracy**: LSTM reduces prediction error by 41% vs moving average
- **Pattern Recognition**: Captures complex non-linear price movements
- **Temporal Dependencies**: Effectively learns from sequential price history
- **Statistical Significance**: Proven superiority through rigorous testing

### Technical Analysis
- **Volatility Handling**: Model adapts to market volatility changes
- **Trend Following**: Captures both short-term and medium-term trends  
- **Noise Reduction**: Filters out market noise better than simple averages
- **Robustness**: Consistent performance across different market conditions

## 🔄 Future Enhancements

### Model Improvements
- **Multi-variate Input**: Include volume, technical indicators
- **Attention Mechanisms**: Transformer-based architectures
- **Ensemble Methods**: Combine multiple LSTM models
- **Hyperparameter Tuning**: Grid search optimization

### Feature Engineering
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Market Sentiment**: News sentiment analysis
- **Economic Indicators**: Interest rates, economic data
- **Cross-Asset Signals**: Market correlation features

## 📁 Repository Structure

```
APPLE-Stock-LSTM-Prediction/
├── APPLE_Stock_Market_Prediction_with_LSTM.ipynb
├── README.md
├── model.h5              # Trained LSTM model
├── scaler.pkl           # Data preprocessing scaler
└── requirements.txt     # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow keras yfinance scikit-learn pandas numpy matplotlib statsmodels
```

### Usage
1. **Clone the repository**
2. **Install dependencies** from requirements.txt
3. **Run the Jupyter notebook** for complete analysis
4. **Load saved model** for inference: `model = load_model('model.h5')`

## 👨‍💻 Author

**Gabriel Ntwari**  
Senior Data Scientist | Ministry of Finance and Economic Planning, Rwanda  
Carnegie Mellon University Africa Alumni

## 📊 Business Applications

This model demonstrates real-world applications in:
- **Algorithmic Trading**: Automated trading signal generation
- **Portfolio Management**: Risk assessment and optimization
- **Financial Planning**: Investment timing decisions
- **Market Research**: Price trend analysis

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Stock market predictions involve significant risk, and past performance does not guarantee future results. Always consult financial professionals before making investment decisions.

## 📄 License

This project is available for educational and research purposes under the MIT License.

## 📞 Connect

- **Email**: ntwarig@alumni.cmu.edu
- **LinkedIn**: [gabriel-ntwari](https://linkedin.com/in/gabriel-ntwari)
- **Portfolio**: [gabriel-ntwari.netlify.app](https://gabriel-ntwari.netlify.app)

---

*Demonstrating the power of deep learning in financial time series forecasting with rigorous statistical validation.*

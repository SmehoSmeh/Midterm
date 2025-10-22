# Autoencoder Anomaly Detection for Market Manipulation

A browser-based JavaScript application that uses MLP (Multi-Layer Perceptron) autoencoders to detect anomalous market conditions in cryptocurrency trading data from Binance API.

## Overview

This application implements an unsupervised learning approach to identify market manipulation patterns such as:
- Liquidation cascades
- Unusual funding spikes
- Abnormal volume patterns
- Price manipulation attempts

The system uses a neural autoencoder trained on "normal" market periods and flags periods with high reconstruction error as potential anomalies.

## Features

### Core Functionality
- **Real-time Data Fetching**: Direct integration with Binance API for live market data
- **Batch Data Processing**: Efficient handling of large date ranges with automatic batching
- **MLP Autoencoder**: Deep neural network architecture for pattern recognition
- **Ensemble Autoencoder**: Multiple model voting system for improved accuracy
- **Advanced Feature Engineering**: 12 comprehensive market features including RSI, Bollinger Bands, and market regime detection

### Data Management
- **Flexible Date Ranges**: Custom date range selection or default lookback periods
- **Multiple Time Intervals**: Support for 1h, 4h, 1d, and 1w intervals
- **Data Quality Analysis**: Automatic detection of missing data, zero volume, and corrupted records
- **Raw Data Export**: Download processed data as JSON files
- **MinMax Normalization**: Consistent data scaling for optimal training

### Anomaly Detection
- **Multi-level Severity Classification**: Normal, Warning, and Critical anomaly levels
- **Sensitive Threshold Detection**: Optimized thresholds for catching major market events
- **Feature Contribution Analysis**: Detailed breakdown of which market features contribute to anomalies
- **Major Event Detection**: Specialized algorithms for identifying crash patterns and market manipulation
- **Historical Event Analysis**: Automatic detection of significant market events

### Visualization & Analysis
- **Interactive Charts**: Training progress, reconstruction errors, and severity distribution
- **Price & Volume Timeline**: Dual-axis charts showing price and volume trends
- **Anomaly Timeline**: Color-coded scatter plots showing anomaly distribution over time
- **Severity Distribution**: Doughnut charts showing anomaly breakdown
- **Feature Contribution Charts**: Visual analysis of feature importance

### User Interface
- **Modern Dark Theme**: Professional gradient-based UI with glassmorphism effects
- **Responsive Design**: Mobile-friendly interface with adaptive layouts
- **Real-time Progress Tracking**: Live progress bars and status updates
- **Interactive Controls**: Customizable training parameters and data range selection
- **Data Sample Tables**: Preview of raw and processed data
- **Statistics Dashboard**: Comprehensive data quality and performance metrics

### Model Management
- **Model Persistence**: Save and load trained models using IndexedDB
- **Configuration Storage**: Persistent storage of model parameters and thresholds
- **Training History**: Track and visualize training progress over epochs
- **Early Stopping**: Automatic training termination to prevent overfitting
- **Memory Management**: Efficient tensor disposal and memory cleanup

## Feature Selection & Engineering

### Current Encoder Features (12 Features)

The autoencoder encoder currently uses 12 carefully selected market features optimized for anomaly detection:

#### **Core Price Features**
1. **`priceChange`** - Close-to-close percentage change
   - Formula: `((close - previous_close) / previous_close) * 100`
   - Purpose: Captures basic price movements and trends

2. **`priceAcceleration`** - Second derivative of price changes
   - Formula: `priceChange - previous_priceChange`
   - Purpose: Detects sudden price accelerations (crash patterns)

3. **`priceGap`** - Gap between opening price and previous close
   - Formula: `|open - previous_close| / previous_close * 100`
   - Purpose: Identifies overnight gaps and flash crashes

4. **`priceMomentum`** - 3-period price momentum
   - Formula: `(close - close_3_periods_ago) / close_3_periods_ago * 100`
   - Purpose: Captures medium-term price trends

#### **Volume Features**
5. **`volume`** - Raw trading volume (MinMax normalized)
   - Purpose: Market activity and liquidity indicator

6. **`volumeSpike`** - Volume spikes above 30% threshold
   - Formula: `max(0, volumeChange - 30)`
   - Purpose: Detects unusual volume surges

7. **`volumeMomentum`** - 3-period volume momentum
   - Formula: `(volume - volume_3_periods_ago) / volume_3_periods_ago * 100`
   - Purpose: Captures volume trend changes

#### **Technical Indicators**
8. **`rsi`** - 14-period Relative Strength Index
   - Range: 0-100 (normalized to 0-1)
   - Purpose: Momentum oscillator for overbought/oversold conditions

9. **`bollingerPosition`** - Position within Bollinger Bands
   - Range: -1 (lower band) to 1 (upper band)
   - Purpose: Volatility and mean reversion indicator

10. **`marketRegime`** - Market trend detection
    - Values: -1 (downtrend), 0 (ranging), 1 (uptrend)
    - Purpose: Contextual market state classification

#### **Simulated Features**
11. **`fundingRateProxy`** - Simulated funding rate for spot markets
    - Formula: `(momentumFactor * 0.5 + volumeFactor * 0.3 + volatilityFactor * 0.2) * 0.01`
    - Purpose: Approximates futures funding rate behavior

12. **`openInterestProxy`** - Simulated open interest
    - Formula: `(volumeFactor * 0.4 + tradeIntensity * 0.3 + priceMomentum * 0.3)`
    - Purpose: Estimates market positioning

### Feature Selection Recommendations

#### **Core Feature Set (8 Features) - Recommended**
For optimal performance with reduced complexity:
```javascript
const coreFeatures = [
    'priceChange',      // Essential price movement
    'volume',           // Market activity
    'priceAcceleration', // Sudden price changes
    'volumeSpike',     // Unusual volume
    'priceGap',        // Price gaps
    'rsi',            // Momentum oscillator
    'bollingerPosition', // Volatility position
    'marketRegime'    // Trend context
];
```

#### **Enhanced Feature Set (10 Features)**
For comprehensive analysis with moderate complexity:
```javascript
const enhancedFeatures = [
    'priceChange',      // Price movement
    'volume',           // Volume
    'priceAcceleration', // Price acceleration
    'volumeSpike',     // Volume spikes
    'priceGap',        // Price gaps
    'priceMomentum',   // Price momentum
    'volumeMomentum',  // Volume momentum
    'rsi',            // RSI
    'bollingerPosition', // Bollinger position
    'marketRegime'    // Market regime
];
```

### Feature Importance Analysis

#### **High Importance Features** ⭐⭐⭐
- **`priceChange`** - Fundamental price movement indicator
- **`priceAcceleration`** - Critical for crash detection
- **`volumeSpike`** - Direct anomaly indicator
- **`priceGap`** - Flash crash detection

#### **Medium Importance Features** ⭐⭐
- **`volume`** - Market activity baseline
- **`rsi`** - Momentum context
- **`bollingerPosition`** - Volatility context
- **`marketRegime`** - Trend context

#### **Lower Importance Features** ⭐
- **`priceMomentum`** - Redundant with priceAcceleration
- **`volumeMomentum`** - Redundant with volumeSpike
- **`fundingRateProxy`** - Limited value for spot markets
- **`openInterestProxy`** - Simulated feature

### Architecture Optimization

When reducing features, adjust the autoencoder architecture:

```javascript
// For 8 features
this.inputSize = 8;
this.encoderUnits = [12, 6, 3];
this.latentSize = 2;
this.decoderUnits = [3, 6, 12];

// For 10 features
this.inputSize = 10;
this.encoderUnits = [14, 7, 3];
this.latentSize = 2;
this.decoderUnits = [3, 7, 14];
```

### Alternative Features to Consider

#### **Volatility Features**
- **`volatility`** - Intraday price range: `((high - low) / close) * 100`
- **`volatilityRatio`** - Current vs average volatility

#### **Volume Features**
- **`volumeRatio`** - Current vs average volume
- **`buySellRatio`** - Taker buy vs sell volume ratio

#### **Price Features**
- **`vwapDeviation`** - Deviation from Volume Weighted Average Price
- **`priceRangePosition`** - Position within daily range: `(close - low) / (high - low)`

### Feature Selection Testing

#### **Ablation Studies**
1. Train models with different feature combinations
2. Compare reconstruction errors and anomaly detection accuracy
3. Analyze feature contribution scores for each anomaly

#### **Cross-Validation**
1. Test on different time periods (bull/bear markets)
2. Validate on different trading pairs
3. Measure performance across various market conditions

#### **Performance Metrics**
- **Reconstruction Error**: Lower is better
- **Anomaly Detection Rate**: Balance between sensitivity and specificity
- **False Positive Rate**: Minimize false alarms
- **Training Time**: Faster training with fewer features

## Architecture

### Data Layer (`binance-data-loader.js`)
- **Batch Data Fetching**: Handles large date ranges with automatic API batching (up to 1000 records per request)
- **Multiple Time Intervals**: Support for 1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M intervals
- **Advanced Feature Engineering**: Calculates 12 comprehensive features:
  - Price change percentage and acceleration
  - Volume changes and spikes
  - Funding rate proxy (simulated for spot markets)
  - Open interest proxy (simulated for spot markets)
  - Price gaps and momentum indicators
  - Volume momentum analysis
  - RSI (Relative Strength Index) calculation
  - Bollinger Bands position
  - Market regime detection (trending vs ranging)
- **Data Quality Control**: Automatic detection and reporting of data quality issues
- **MinMax Normalization**: Consistent scaling across all features for optimal training
- **Historical Event Analysis**: Built-in detection of major market events and crash patterns

### Model Layer (`autoencoder.js`)
- **MLP Autoencoder**: Multi-layer perceptron with encoder-decoder architecture
  - **Encoder**: 12 → 16 → 8 → 4 → 2 neurons (compression to latent space)
  - **Decoder**: 2 → 4 → 8 → 16 → 12 neurons (reconstruction)
  - **Dropout**: 0.2 dropout rate for regularization
  - **Activation**: ReLU for hidden layers, linear for output
- **Ensemble Autoencoder**: Multiple model voting system for improved accuracy
  - 3 different autoencoder configurations
  - Majority voting for anomaly classification
  - Confidence scoring for ensemble predictions
- **Training Optimization**: 
  - Adam optimizer with learning rate 0.001
  - Mean Squared Error loss function
  - Early stopping with patience of 5 epochs
  - Batch size optimization (default 64)
- **Anomaly Detection**: 
  - Statistical threshold calculation (mean + 1.5 × std)
  - Multi-level severity classification (Normal/Warning/Critical)
  - Feature contribution analysis for root cause identification
  - Major event pattern recognition

### Application Layer (`app.js`)
- **UI Management**: Comprehensive interface for all user interactions
- **Workflow Coordination**: Orchestrates data fetching, training, and detection processes
- **Real-time Visualization**: 
  - Training loss charts with validation curves
  - Reconstruction error timeline with color-coded severity
  - Anomaly severity distribution (doughnut charts)
  - Price and volume timeline with dual-axis display
- **Data Visualization**: 
  - Raw data sample tables
  - Comprehensive statistics dashboard
  - Data quality metrics and warnings
  - Historical events analysis with October 10th special detection
- **Model Persistence**: Save/load functionality using IndexedDB and localStorage
- **Memory Management**: Proper tensor disposal and resource cleanup

## Usage Instructions

### 1. Setup
1. Open `index.html` in a modern web browser
2. Ensure internet connection for Binance API access
3. No additional installation required (uses CDN libraries)

### 2. Data Collection
1. Select a trading pair from the dropdown (BTC/USDT, ETH/USDT, etc.)
2. Choose data interval (1h, 4h, 1d, 1w)
3. Configure date range:
   - **Default**: Use lookback days (default: 60 days)
   - **Custom**: Enable custom range and select start/end dates
4. Click "Fetch Market Data" to download historical data
5. Monitor batch processing progress for large date ranges
6. Review comprehensive data summary and quality metrics
7. Examine data sample table and statistics dashboard

### 3. Model Training
1. Adjust training parameters if needed:
   - **Epochs**: Number of training iterations (default: 30)
   - **Batch Size**: Training batch size (default: 64)
2. Click "Train Autoencoder" to begin training
3. Monitor real-time training progress and loss curves
4. Early stopping will automatically terminate training if no improvement
5. Training typically takes 1-3 minutes for standard model
6. Review training history and model performance metrics

### 4. Anomaly Detection
1. Click "Detect Anomalies" after training completes
2. Review the reconstruction error timeline chart with color-coded severity
3. Examine the anomaly severity distribution (doughnut chart)
4. Check the top anomalies table for specific timestamps and importance scores
5. Analyze feature contributions for the most significant anomalies
6. Review historical events analysis for major market events
7. Examine price and volume timeline for context

### 5. Model Management
- **Save Model**: Store trained weights and configuration locally using IndexedDB
- **Load Model**: Restore previously saved model and configuration
- **Download Data**: Export raw market data as JSON file
- **Reset**: Clear all data and start fresh with clean state

## API Integration

### Binance API Endpoints Used
- **Klines**: `https://api.binance.com/api/v3/klines`
  - Parameters: `symbol`, `interval`, `startTime`, `endTime`, `limit`
  - Returns: OHLCV data with comprehensive market information
  - Rate Limiting: 100ms delay between batch requests
  - Maximum Records: 1000 per request (Binance API limit)

### Data Processing Pipeline
The application processes raw Binance data into twelve normalized features:

1. **Price Change**: Close-to-close percentage change
2. **Volume**: Normalized trading volume
3. **Funding Rate Proxy**: Simulated funding rate using price momentum and volume
4. **Open Interest Proxy**: Simulated open interest using volume and trade activity
5. **Price Acceleration**: Second derivative of price changes
6. **Volume Spike**: Detection of unusual volume increases (>30%)
7. **Price Gap**: Gap between opening price and previous close
8. **Price Momentum**: 3-period price momentum indicator
9. **Volume Momentum**: 3-period volume momentum indicator
10. **RSI**: 14-period Relative Strength Index
11. **Bollinger Position**: Position within Bollinger Bands (-1 to 1)
12. **Market Regime**: Market trend detection (uptrend/ranging/downtrend)

## Anomaly Detection Logic

### Threshold Calculation
- Training data reconstruction errors are analyzed using statistical methods
- Primary threshold: Mean + 1.5 × Standard Deviation
- Percentile-based threshold: 95th percentile of reconstruction errors
- Final threshold: Minimum of statistical and percentile methods for sensitivity
- This captures ~95% of normal market behavior while detecting major events

### Severity Levels
- **Green (Normal)**: Error < threshold
- **Yellow (Warning)**: threshold ≤ error < 1.2 × threshold
- **Red (Critical)**: error ≥ 1.2 × threshold

### Feature Contribution Analysis
For each anomaly, the system calculates which features contributed most to the reconstruction error, helping identify the root cause of unusual market behavior. The analysis includes:

- **Individual Feature Contributions**: Absolute difference between original and reconstructed values
- **Feature Importance Ranking**: Sorted by contribution magnitude
- **Pattern Recognition**: Detection of specific anomaly patterns (crashes, spikes, etc.)
- **Major Event Classification**: Special flags for significant market events

### Major Event Detection
The system includes specialized algorithms for detecting major market events:

- **Crash Patterns**: Price drop + volume spike combinations
- **Flash Crashes**: Extreme negative price changes with massive volume
- **Market Manipulation**: Unusual volume spikes with minimal price change
- **Liquidation Cascades**: Multiple consecutive anomalies with increasing severity

## Interpretation Guide

### Understanding Results

#### Reconstruction Error Timeline
- **X-axis**: Time points (chronological order)
- **Y-axis**: Reconstruction error magnitude
- **Colors**: 
  - Green dots: Normal market conditions
  - Yellow dots: Elevated risk periods
  - Red dots: Critical anomalies
- **Gray line**: Anomaly threshold

#### Severity Distribution
- Shows percentage breakdown of normal vs. anomalous periods
- Helps assess overall market stability
- Higher anomaly rates may indicate volatile market conditions

#### Top Anomalies Table
- Lists timestamps of most significant anomalies
- Shows exact reconstruction error values
- Provides severity classification
- Enables correlation with external market events

#### Feature Contribution Analysis
- Identifies which market features drove the anomaly
- **Price Change**: Unusual price movements
- **Volume**: Abnormal trading volume
- **Funding Rate Proxy**: Momentum-based anomalies
- **Open Interest Proxy**: Activity-based anomalies

### Common Anomaly Patterns

#### Liquidation Cascades
- High volume with negative price change
- Elevated funding rate proxy
- Multiple consecutive anomalies

#### Market Manipulation
- Unusual volume spikes with minimal price change
- Abnormal funding rate patterns
- Sudden reversals in price direction

#### Flash Crashes
- Extreme negative price changes
- Massive volume spikes
- Critical severity classification

## Technical Specifications

### Dependencies
- **TensorFlow.js**: 4.22.0 (neural network framework)
- **Chart.js**: 4.4.1 (data visualization)
- **Native Fetch API**: Binance API communication
- **IndexedDB**: Model persistence storage
- **LocalStorage**: Configuration and metadata storage

### Browser Requirements
- Modern browser with ES6+ support
- WebGL support for TensorFlow.js GPU acceleration
- IndexedDB support for model persistence
- Minimum 4GB RAM recommended
- Chrome 80+, Firefox 75+, Safari 13+, Edge 80+

### Performance Specifications
- **Data Fetching**: 
  - Single request: ~2-5 seconds (up to 1000 records)
  - Batch processing: ~10-30 seconds (large date ranges)
  - Rate limiting: 100ms delay between batches
- **Training Performance**:
  - Standard model: 1-3 minutes (30 epochs)
  - Ensemble model: 3-9 minutes (3 models × 30 epochs)
  - Early stopping: Automatic termination after 5 epochs without improvement
- **Anomaly Detection**: <1 second for validation data
- **Memory Usage**: 
  - During training: ~100-200MB
  - During inference: ~50-100MB
  - Tensor cleanup: Automatic disposal after operations

### Data Processing Capabilities
- **Maximum Data Points**: 1000 per API request (Binance limit)
- **Batch Processing**: Automatic splitting for large date ranges
- **Feature Engineering**: 12 comprehensive market indicators
- **Normalization**: MinMax scaling across all features
- **Data Quality**: Automatic validation and error reporting

### Model Architecture Details
- **Input Features**: 12-dimensional feature vectors
- **Latent Space**: 2-dimensional compressed representation
- **Network Depth**: 5 layers (encoder) + 5 layers (decoder)
- **Total Parameters**: ~1,000 trainable parameters
- **Activation Functions**: ReLU (hidden), Linear (output)
- **Regularization**: Dropout (0.2) and early stopping

## User Interface Features

### Modern Design Elements
- **Glassmorphism Effects**: Frosted glass appearance with backdrop blur
- **Gradient Backgrounds**: Professional blue-purple gradient theme
- **Responsive Grid Layout**: Adaptive design for all screen sizes
- **Interactive Animations**: Smooth hover effects and transitions
- **Color-coded Elements**: Consistent color palette for different data types

### Advanced Controls
- **Custom Date Range Selection**: DateTime picker with validation
- **Flexible Time Intervals**: Support for multiple trading timeframes
- **Training Parameter Adjustment**: Real-time modification of epochs and batch size
- **Data Range Options**: Choice between custom dates or default lookback periods
- **Model Management**: Save/load functionality with persistent storage

### Data Visualization Dashboard
- **Real-time Progress Tracking**: Live progress bars with percentage indicators
- **Interactive Charts**: 
  - Training loss curves with validation metrics
  - Reconstruction error scatter plots with severity coloring
  - Anomaly distribution doughnut charts
  - Price/volume dual-axis timeline charts
- **Data Quality Metrics**: Comprehensive statistics with color-coded indicators
- **Sample Data Tables**: Preview of raw and processed market data
- **Historical Events Analysis**: Special detection for significant market events

### User Experience Enhancements
- **Status Messages**: Real-time feedback on all operations
- **Error Handling**: User-friendly error messages with troubleshooting hints
- **Loading States**: Visual indicators during data processing and training
- **Button State Management**: Context-aware enabling/disabling of controls
- **Memory Management**: Automatic cleanup and resource optimization

## Troubleshooting

### Common Issues

#### "Failed to fetch market data"
- Check internet connection
- Verify Binance API is accessible
- Try different trading pair
- Check browser console for detailed errors

#### "Training failed"
- Ensure sufficient system memory
- Try reducing batch size or epochs
- Close other browser tabs
- Restart browser if memory issues persist

#### "No anomalies detected"
- Normal market conditions (good sign!)
- Try different time periods
- Adjust threshold sensitivity in code
- Check if training completed successfully

#### Charts not displaying
- Ensure Chart.js loaded correctly
- Check browser console for JavaScript errors
- Try refreshing the page
- Verify WebGL support

### Performance Optimization
- Use smaller batch sizes on slower devices
- Reduce training epochs for faster results
- Close unnecessary browser tabs
- Use Chrome/Firefox for best performance

## Future Enhancements

### Planned Features
- **Real-time Monitoring Mode**: Live data streaming with WebSocket integration
- **Multiple Timeframe Analysis**: Simultaneous analysis across different intervals
- **Custom Threshold Adjustment**: User-configurable anomaly detection sensitivity
- **Advanced Export Functionality**: CSV, JSON, and PDF report generation
- **Additional Trading Pairs**: Support for more cryptocurrency pairs
- **Advanced Feature Engineering**: Technical indicators like MACD, Stochastic, etc.
- **Model Comparison Tools**: Side-by-side comparison of different model architectures

### Technical Improvements
- **WebSocket Integration**: Real-time data streaming from Binance
- **Model Ensemble Methods**: Advanced voting and stacking techniques
- **Automated Hyperparameter Tuning**: Grid search and Bayesian optimization
- **Cloud Deployment Options**: Docker containers and cloud hosting
- **Mobile-responsive Design**: Enhanced mobile experience
- **Progressive Web App**: Offline functionality and app-like experience
- **API Integration**: RESTful API for external system integration

### Advanced Analytics
- **Pattern Recognition**: Machine learning-based pattern identification
- **Market Sentiment Analysis**: Integration with social media and news data
- **Risk Assessment**: Portfolio-level risk analysis
- **Backtesting Framework**: Historical performance validation
- **Alert System**: Email and push notifications for anomalies
- **Custom Dashboards**: User-configurable visualization layouts

## Contributing

This is an educational project demonstrating autoencoder-based anomaly detection in financial markets. Contributions and improvements are welcome.

### Development Setup
1. Clone the repository
2. Open `index.html` in a local web server
3. Modify code as needed
4. Test thoroughly before deployment

## License

This project is for educational purposes. Please ensure compliance with Binance API terms of service when using their data endpoints.

## References

- [Binance API Documentation](https://developers.binance.com/docs/binance-spot-api-docs/rest-api#market-data-endpoints)
- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [Chart.js Documentation](https://www.chartjs.org/docs/)
- Autoencoder Theory: Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks.

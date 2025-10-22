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

- **Real-time Data Fetching**: Direct integration with Binance API for live market data
- **MLP Autoencoder**: Deep neural network architecture for pattern recognition
- **Visual Anomaly Detection**: Color-coded visualization of anomaly severity levels
- **Feature Analysis**: Detailed breakdown of which market features contribute to anomalies
- **Model Persistence**: Save and load trained models for future use
- **Interactive Charts**: Training progress, reconstruction errors, and severity distribution

## Architecture

### Data Layer (`binance-data-loader.js`)
- Fetches 1-hour candlestick data from Binance API
- Processes 30 days of historical data (720 data points)
- Calculates derived features:
  - Price change percentage
  - Volume changes
  - Funding rate proxy (for spot markets)
  - Open interest proxy
- Applies MinMax normalization for consistent training

### Model Layer (`autoencoder.js`)
- **Encoder**: 4 → 16 → 8 → 4 → 2 neurons (compression)
- **Decoder**: 2 → 4 → 8 → 16 → 4 neurons (reconstruction)
- **Training**: Mean Squared Error loss with Adam optimizer
- **Anomaly Detection**: Threshold-based classification using reconstruction error

### Application Layer (`app.js`)
- Manages UI interactions and workflow
- Coordinates data fetching, training, and detection
- Handles visualization and result display
- Provides model persistence functionality

## Usage Instructions

### 1. Setup
1. Open `index.html` in a modern web browser
2. Ensure internet connection for Binance API access
3. No additional installation required (uses CDN libraries)

### 2. Data Collection
1. Select a trading pair from the dropdown (BTC/USDT, ETH/USDT, etc.)
2. Click "Fetch Market Data" to download 30 days of historical data
3. Wait for data processing and normalization to complete
4. Review the data summary showing training/validation split

### 3. Model Training
1. Adjust training parameters if needed:
   - **Epochs**: Number of training iterations (default: 100)
   - **Batch Size**: Training batch size (default: 32)
2. Click "Train Autoencoder" to begin training
3. Monitor training progress and loss curves
4. Training typically takes 1-3 minutes depending on hardware

### 4. Anomaly Detection
1. Click "Detect Anomalies" after training completes
2. Review the reconstruction error timeline chart
3. Examine the anomaly severity distribution
4. Check the top anomalies table for specific timestamps
5. Analyze feature contributions for the most significant anomalies

### 5. Model Management
- **Save Model**: Store trained weights and configuration locally
- **Load Model**: Restore previously saved model
- **Reset**: Clear all data and start fresh

## API Integration

### Binance API Endpoints Used
- **Klines**: `https://api.binance.com/api/v3/klines`
  - Parameters: `symbol`, `interval=1h`, `limit=720`
  - Returns: OHLCV data for 30 days

### Data Processing
The application processes raw Binance data into four normalized features:

1. **Price Change**: Close-to-close percentage change
2. **Volume**: Normalized trading volume
3. **Funding Rate Proxy**: Simulated funding rate using price momentum and volume
4. **Open Interest Proxy**: Simulated open interest using volume and trade activity

## Anomaly Detection Logic

### Threshold Calculation
- Training data reconstruction errors are analyzed
- Threshold = Mean + 2.5 × Standard Deviation
- This captures ~99% of normal market behavior

### Severity Levels
- **Green (Normal)**: Error < threshold
- **Yellow (Warning)**: threshold ≤ error < 1.5 × threshold
- **Red (Critical)**: error ≥ 1.5 × threshold

### Feature Contribution Analysis
For each anomaly, the system calculates which features contributed most to the reconstruction error, helping identify the root cause of unusual market behavior.

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

### Browser Requirements
- Modern browser with ES6+ support
- WebGL support for TensorFlow.js
- IndexedDB support for model persistence
- Minimum 4GB RAM recommended

### Performance
- **Data Fetching**: ~2-5 seconds
- **Training Time**: 1-3 minutes (100 epochs)
- **Anomaly Detection**: <1 second
- **Memory Usage**: ~50-100MB during training

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
- Real-time monitoring mode
- Multiple timeframe analysis
- Custom threshold adjustment
- Export functionality for results
- Additional trading pairs
- Advanced feature engineering

### Technical Improvements
- WebSocket integration for live data
- Model ensemble methods
- Automated hyperparameter tuning
- Cloud deployment options
- Mobile-responsive design

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

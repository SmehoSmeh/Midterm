/**
 * BinanceDataLoader class for fetching and preprocessing cryptocurrency market data
 * from Binance API for autoencoder anomaly detection
 */
class BinanceDataLoader {
    constructor() {
        this.baseUrl = 'https://api.binance.com';
        this.rawData = null;
        this.processedData = null;
        this.normalizedData = null;
        this.minMaxValues = {};
        this.symbol = 'BTCUSDT';
        
        // Batch fetching configuration
        this.maxApiLimit = 1000; // Binance API maximum limit
        this.batchSize = 900; // Safe margin below limit
        this.marginError = 100; // Safety margin for calculations
        
        // Date range configuration (user configurable)
        this.startDate = null; // Will be set by user or default to 60 days ago
        this.endDate = null; // Will be set by user or default to now
        
        // Configuration - Optimized training period
        this.lookbackDays = 60; // 2 months of data (reduced for faster training)
        this.interval = '1h';
        this.limit = 1500; // 60 days * 24 hours
        
        // Training/validation split - Percentage-based for flexible data ranges
        this.trainPercentage = 0.8; // 80% for training
        this.validationPercentage = 0.2; // 20% for validation
        
        // Legacy configuration (kept for backward compatibility)
        this.trainDays = 50; // ~1.7 months training
        this.validationDays = 10; // ~1.3 weeks validation
        
        // Enhanced configuration
        this.minTrainingSamples = 1500; // Minimum samples for reliable training
        this.outlierThreshold = 3.0; // Z-score threshold for outlier detection
    }

    /**
     * Fetch market data from Binance API
     * @param {string} symbol - Trading pair symbol (e.g., 'BTCUSDT')
     * @returns {Promise<Object>} Raw market data
     */
    async fetchMarketData(symbol = 'BTCUSDT') {
        this.symbol = symbol;
        this.updateStatus('Fetching market data from Binance...');
        
        try {
            // Calculate date range for better debugging
            const endTime = Date.now();
            const startTime = endTime - (this.lookbackDays * 24 * 60 * 60 * 1000);
            const startDate = new Date(startTime);
            const endDate = new Date(endTime);
            
            console.log(`Fetching data from ${startDate.toISOString()} to ${endDate.toISOString()}`);
            console.log(`Looking for October 10th events in range: ${startDate.toDateString()} - ${endDate.toDateString()}`);
            
            // Fetch kline/candlestick data with explicit date range
            const klineUrl = `${this.baseUrl}/api/v3/klines?symbol=${symbol}&interval=${this.interval}&startTime=${startTime}&endTime=${endTime}&limit=${this.limit}`;
            const response = await fetch(klineUrl);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const klineData = await response.json();
            
            // Process kline data into structured format
            this.rawData = this.processKlineData(klineData);
            
            this.updateStatus(`Successfully fetched ${this.rawData.length} data points for ${symbol}`);
            return this.rawData;
            
        } catch (error) {
            throw new Error(`Failed to fetch market data: ${error.message}`);
        }
    }

    /**
     * Set custom date range for data fetching
     * @param {Date|string} startDate - Start date (Date object or ISO string)
     * @param {Date|string} endDate - End date (Date object or ISO string)
     */
    setDateRange(startDate, endDate) {
        this.startDate = startDate instanceof Date ? startDate : new Date(startDate);
        this.endDate = endDate instanceof Date ? endDate : new Date(endDate);
        
        // Validate date range
        if (this.startDate >= this.endDate) {
            throw new Error('Start date must be before end date');
        }
        
        console.log(`Date range set: ${this.startDate.toISOString()} to ${this.endDate.toISOString()}`);
    }

    /**
     * Calculate optimal batch configuration for large data ranges
     * @param {Date} startDate - Start date
     * @param {Date} endDate - End date
     * @param {string} interval - Kline interval
     * @returns {Object} Batch configuration
     */
    calculateBatchConfig(startDate, endDate, interval = '1h') {
        const totalTimeMs = endDate.getTime() - startDate.getTime();
        
        // Calculate interval in milliseconds
        const intervalMs = this.getIntervalMs(interval);
        
        // Calculate total data points needed
        const totalDataPoints = Math.ceil(totalTimeMs / intervalMs);
        
        // Calculate number of batches needed
        const batchesNeeded = Math.ceil(totalDataPoints / this.batchSize);
        
        // Calculate time per batch
        const timePerBatch = totalTimeMs / batchesNeeded;
        
        console.log(`Batch configuration: ${totalDataPoints} total points, ${batchesNeeded} batches, ${timePerBatch/1000/60/60} hours per batch`);
        
        return {
            totalDataPoints,
            batchesNeeded,
            timePerBatch,
            intervalMs
        };
    }

    /**
     * Convert interval string to milliseconds
     * @param {string} interval - Interval string (e.g., '1h', '1d', '1m')
     * @returns {number} Milliseconds
     */
    getIntervalMs(interval) {
        const intervalMap = {
            '1s': 1000,
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000
        };
        
        return intervalMap[interval] || intervalMap['1h'];
    }

    /**
     * Fetch market data in batches for large date ranges
     * @param {string} symbol - Trading pair symbol
     * @param {Date} startDate - Start date
     * @param {Date} endDate - End date
     * @param {string} interval - Kline interval
     * @param {Function} progressCallback - Progress callback function
     * @returns {Promise<Array>} Combined raw market data
     */
    async fetchMarketDataBatched(symbol = 'BTCUSDT', startDate = null, endDate = null, interval = '1h', progressCallback = null) {
        this.symbol = symbol;
        
        // Use provided dates or defaults
        const start = startDate || this.startDate || new Date(Date.now() - (this.lookbackDays * 24 * 60 * 60 * 1000));
        const end = endDate || this.endDate || new Date();
        
        this.updateStatus(`Fetching market data from ${start.toISOString()} to ${end.toISOString()}...`);
        
        try {
            // Calculate batch configuration
            const batchConfig = this.calculateBatchConfig(start, end, interval);
            
            if (batchConfig.totalDataPoints <= this.maxApiLimit) {
                // Single request if within limit
                console.log('Data range within API limit, using single request');
                return await this.fetchMarketData(symbol);
            }
            
            // Batch fetching for large ranges
            console.log(`Using batch fetching: ${batchConfig.batchesNeeded} batches`);
            const allData = [];
            
            for (let i = 0; i < batchConfig.batchesNeeded; i++) {
                const batchStartTime = start.getTime() + (i * batchConfig.timePerBatch);
                const batchEndTime = Math.min(
                    start.getTime() + ((i + 1) * batchConfig.timePerBatch),
                    end.getTime()
                );
                
                const batchStart = new Date(batchStartTime);
                const batchEnd = new Date(batchEndTime);
                
                console.log(`Fetching batch ${i + 1}/${batchConfig.batchesNeeded}: ${batchStart.toISOString()} to ${batchEnd.toISOString()}`);
                
                // Update progress
                if (progressCallback) {
                    progressCallback(i + 1, batchConfig.batchesNeeded, `Fetching batch ${i + 1}/${batchConfig.batchesNeeded}`);
                }
                
                // Fetch batch data
                const batchUrl = `${this.baseUrl}/api/v3/klines?symbol=${symbol}&interval=${interval}&startTime=${batchStartTime}&endTime=${batchEndTime}&limit=${this.batchSize}`;
                const response = await fetch(batchUrl);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const batchData = await response.json();
                allData.push(...batchData);
                
                // Add small delay to avoid rate limiting
                if (i < batchConfig.batchesNeeded - 1) {
                    await new Promise(resolve => setTimeout(resolve, 100));
                }
            }
            
            // Process combined data
            this.rawData = this.processKlineData(allData);
            
            this.updateStatus(`Successfully fetched ${this.rawData.length} data points across ${batchConfig.batchesNeeded} batches`);
            return this.rawData;
            
        } catch (error) {
            throw new Error(`Failed to fetch batched market data: ${error.message}`);
        }
    }

    /**
     * Download raw data as JSON file
     * @param {string} filename - Optional filename
     */
    downloadRawData(filename = null) {
        if (!this.rawData || this.rawData.length === 0) {
            throw new Error('No raw data available to download');
        }
        
        const defaultFilename = `${this.symbol}_raw_data_${new Date().toISOString().split('T')[0]}.json`;
        const downloadFilename = filename || defaultFilename;
        
        const dataStr = JSON.stringify(this.rawData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = downloadFilename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        console.log(`Raw data downloaded: ${downloadFilename}`);
    }

    /**
     * Process raw kline data into structured format
     * @param {Array} klineData - Raw kline data from Binance API
     * @returns {Array} Processed data points
     */
    processKlineData(klineData) {
        const processedData = [];
        
        for (let i = 0; i < klineData.length; i++) {
            const kline = klineData[i];
            
            // Extract features from kline data
            const timestamp = parseInt(kline[0]);
            const open = parseFloat(kline[1]);
            const high = parseFloat(kline[2]);
            const low = parseFloat(kline[3]);
            const close = parseFloat(kline[4]);
            const volume = parseFloat(kline[5]);
            const closeTime = parseInt(kline[6]);
            const quoteVolume = parseFloat(kline[7]);
            const trades = parseInt(kline[8]);
            const takerBuyBaseVolume = parseFloat(kline[9]);
            const takerBuyQuoteVolume = parseFloat(kline[10]);
            
            // Calculate derived features with enhanced anomaly detection
            const priceChange = i > 0 ? ((close - processedData[i-1].close) / processedData[i-1].close) * 100 : 0;
            const volumeChange = i > 0 ? ((volume - processedData[i-1].volume) / processedData[i-1].volume) * 100 : 0;
            const volatility = high > low ? ((high - low) / close) * 100 : 0;
            
            // Enhanced features for better anomaly detection
            const priceAcceleration = i > 1 ? priceChange - ((processedData[i-1].close - processedData[i-2].close) / processedData[i-2].close) * 100 : 0;
            const volumeSpike = i > 0 ? Math.max(0, volumeChange - 30) : 0; // Detect volume spikes > 30% (more sensitive)
            const priceGap = Math.abs(open - processedData[i-1]?.close || open) / (processedData[i-1]?.close || open) * 100;
            
            // Advanced features for enhanced accuracy
            const priceMomentum = i > 2 ? (close - processedData[i-3].close) / processedData[i-3].close * 100 : 0;
            const volumeMomentum = i > 2 ? (volume - processedData[i-3].volume) / processedData[i-3].volume * 100 : 0;
            const rsi = this.calculateRSI(processedData.slice(Math.max(0, i-13), i+1)); // 14-period RSI
            const bollingerPosition = this.calculateBollingerPosition(processedData.slice(Math.max(0, i-19), i+1), close);
            const marketRegime = this.detectMarketRegime(processedData.slice(Math.max(0, i-23), i+1));
            
            // Calculate funding rate proxy (using price momentum and volume)
            const fundingRateProxy = this.calculateFundingRateProxy(priceChange, volumeChange, volatility);
            
            // Calculate open interest proxy (using volume and price action)
            const openInterestProxy = this.calculateOpenInterestProxy(volume, priceChange, trades);
            
            processedData.push({
                timestamp,
                datetime: new Date(timestamp).toISOString(),
                open,
                high,
                low,
                close,
                volume,
                quoteVolume,
                trades,
                priceChange,
                volumeChange,
                volatility,
                priceAcceleration,
                volumeSpike,
                priceGap,
                priceMomentum,
                volumeMomentum,
                rsi,
                bollingerPosition,
                marketRegime,
                fundingRateProxy,
                openInterestProxy
            });
        }
        
        return processedData;
    }

    /**
     * Calculate funding rate proxy for spot markets
     * @param {number} priceChange - Price change percentage
     * @param {number} volumeChange - Volume change percentage
     * @param {number} volatility - Price volatility
     * @returns {number} Funding rate proxy
     */
    calculateFundingRateProxy(priceChange, volumeChange, volatility) {
        // Simulate funding rate using price momentum and volume patterns
        // Higher positive price change with high volume = positive funding
        // Higher negative price change with high volume = negative funding
        const momentumFactor = Math.tanh(priceChange / 10); // Normalize price change
        const volumeFactor = Math.tanh(volumeChange / 50); // Normalize volume change
        const volatilityFactor = Math.tanh(volatility / 5); // Normalize volatility
        
        return (momentumFactor * 0.5 + volumeFactor * 0.3 + volatilityFactor * 0.2) * 0.01;
    }

    /**
     * Calculate open interest proxy for spot markets
     * @param {number} volume - Trading volume
     * @param {number} priceChange - Price change percentage
     * @param {number} trades - Number of trades
     * @returns {number} Open interest proxy
     */
    calculateOpenInterestProxy(volume, priceChange, trades) {
        // Simulate open interest using volume patterns and trade activity
        const volumeFactor = Math.log(volume + 1) / 10; // Log scale volume
        const tradeIntensity = trades / 1000; // Normalize trade count
        const priceMomentum = Math.abs(priceChange) / 10; // Absolute price change
        
        return (volumeFactor * 0.4 + tradeIntensity * 0.3 + priceMomentum * 0.3);
    }

    /**
     * Calculate RSI (Relative Strength Index)
     * @param {Array} data - Array of price data points
     * @returns {number} RSI value (0-100)
     */
    calculateRSI(data) {
        if (data.length < 2) return 50; // Neutral RSI
        
        let gains = 0;
        let losses = 0;
        
        for (let i = 1; i < data.length; i++) {
            const change = data[i].close - data[i-1].close;
            if (change > 0) gains += change;
            else losses -= change;
        }
        
        if (losses === 0) return 100;
        if (gains === 0) return 0;
        
        const rs = gains / losses;
        return 100 - (100 / (1 + rs));
    }

    /**
     * Calculate Bollinger Bands position
     * @param {Array} data - Array of price data points
     * @param {number} currentPrice - Current price
     * @returns {number} Position within Bollinger Bands (-1 to 1)
     */
    calculateBollingerPosition(data, currentPrice) {
        if (data.length < 20) return 0; // Neutral position
        
        const closes = data.map(d => d.close);
        const sma = closes.reduce((sum, price) => sum + price, 0) / closes.length;
        
        const variance = closes.reduce((sum, price) => sum + Math.pow(price - sma, 2), 0) / closes.length;
        const std = Math.sqrt(variance);
        
        const upperBand = sma + (2 * std);
        const lowerBand = sma - (2 * std);
        
        // Return position: -1 (lower band) to 1 (upper band)
        if (currentPrice <= lowerBand) return -1;
        if (currentPrice >= upperBand) return 1;
        
        return (currentPrice - sma) / (upperBand - sma);
    }

    /**
     * Detect market regime (trending vs ranging)
     * @param {Array} data - Array of price data points
     * @returns {number} Market regime indicator (-1: downtrend, 0: ranging, 1: uptrend)
     */
    detectMarketRegime(data) {
        if (data.length < 24) return 0; // Neutral
        
        const closes = data.map(d => d.close);
        const firstPrice = closes[0];
        const lastPrice = closes[closes.length - 1];
        const priceChange = (lastPrice - firstPrice) / firstPrice * 100;
        
        // Calculate volatility
        const returns = [];
        for (let i = 1; i < closes.length; i++) {
            returns.push((closes[i] - closes[i-1]) / closes[i-1]);
        }
        const volatility = Math.sqrt(returns.reduce((sum, ret) => sum + ret * ret, 0) / returns.length) * 100;
        
        // Regime detection based on trend strength vs volatility
        if (Math.abs(priceChange) > volatility * 2) {
            return priceChange > 0 ? 1 : -1; // Strong trend
        }
        return 0; // Ranging market
    }

    /**
     * Normalize data using MinMax scaling
     */
    normalizeData() {
        if (!this.rawData || this.rawData.length === 0) {
            throw new Error('No raw data available for normalization');
        }

        this.processedData = [...this.rawData];
        this.normalizedData = [];
        this.minMaxValues = {};

        // Calculate min-max values for each feature (enhanced for better anomaly detection)
        const features = [
            'priceChange', 'volume', 'fundingRateProxy', 'openInterestProxy', 
            'priceAcceleration', 'volumeSpike', 'priceGap', 'priceMomentum', 
            'volumeMomentum', 'rsi', 'bollingerPosition', 'marketRegime'
        ];
        
        features.forEach(feature => {
            this.minMaxValues[feature] = {
                min: Math.min(...this.processedData.map(d => d[feature])),
                max: Math.max(...this.processedData.map(d => d[feature]))
            };
        });

        // Apply MinMax normalization
        this.processedData.forEach(dataPoint => {
            const normalizedPoint = {
                timestamp: dataPoint.timestamp,
                datetime: dataPoint.datetime,
                priceChange: this.normalizeValue(dataPoint.priceChange, 'priceChange'),
                volume: this.normalizeValue(dataPoint.volume, 'volume'),
                fundingRateProxy: this.normalizeValue(dataPoint.fundingRateProxy, 'fundingRateProxy'),
                openInterestProxy: this.normalizeValue(dataPoint.openInterestProxy, 'openInterestProxy'),
                priceAcceleration: this.normalizeValue(dataPoint.priceAcceleration, 'priceAcceleration'),
                volumeSpike: this.normalizeValue(dataPoint.volumeSpike, 'volumeSpike'),
                priceGap: this.normalizeValue(dataPoint.priceGap, 'priceGap'),
                priceMomentum: this.normalizeValue(dataPoint.priceMomentum, 'priceMomentum'),
                volumeMomentum: this.normalizeValue(dataPoint.volumeMomentum, 'volumeMomentum'),
                rsi: this.normalizeValue(dataPoint.rsi, 'rsi'),
                bollingerPosition: this.normalizeValue(dataPoint.bollingerPosition, 'bollingerPosition'),
                marketRegime: this.normalizeValue(dataPoint.marketRegime, 'marketRegime'),
                // Keep original values for reference
                original: {
                    priceChange: dataPoint.priceChange,
                    volume: dataPoint.volume,
                    fundingRateProxy: dataPoint.fundingRateProxy,
                    openInterestProxy: dataPoint.openInterestProxy,
                    priceAcceleration: dataPoint.priceAcceleration,
                    volumeSpike: dataPoint.volumeSpike,
                    priceGap: dataPoint.priceGap,
                    priceMomentum: dataPoint.priceMomentum,
                    volumeMomentum: dataPoint.volumeMomentum,
                    rsi: dataPoint.rsi,
                    bollingerPosition: dataPoint.bollingerPosition,
                    marketRegime: dataPoint.marketRegime
                }
            };
            
            this.normalizedData.push(normalizedPoint);
        });

        console.log('Data normalized using MinMax scaling');
        console.log('MinMax values:', this.minMaxValues);
    }

    /**
     * Normalize a single value using MinMax scaling
     * @param {number} value - Value to normalize
     * @param {string} feature - Feature name
     * @returns {number} Normalized value
     */
    normalizeValue(value, feature) {
        const { min, max } = this.minMaxValues[feature];
        const range = max - min;
        return range > 0 ? (value - min) / range : 0;
    }

    /**
     * Denormalize a normalized value back to original scale
     * @param {number} normalizedValue - Normalized value
     * @param {string} feature - Feature name
     * @returns {number} Original scale value
     */
    denormalizeValue(normalizedValue, feature) {
        const { min, max } = this.minMaxValues[feature];
        const range = max - min;
        return normalizedValue * range + min;
    }

    /**
     * Prepare data for autoencoder training
     * @returns {Object} Training and validation data
     */
    prepareTrainingData() {
        if (!this.normalizedData || this.normalizedData.length === 0) {
            throw new Error('No normalized data available');
        }

        // Split data chronologically using percentage-based approach
        const totalPoints = this.normalizedData.length;
        const trainPoints = Math.floor(totalPoints * this.trainPercentage);
        
        console.log(`Data split: ${totalPoints} total points, ${trainPoints} training (${(this.trainPercentage * 100).toFixed(1)}%), ${totalPoints - trainPoints} validation (${(this.validationPercentage * 100).toFixed(1)}%)`);
        
        const trainData = this.normalizedData.slice(0, trainPoints);
        const validationData = this.normalizedData.slice(trainPoints);

        // Extract feature vectors for training (enhanced with 12 features)
        const trainFeatures = trainData.map(point => [
            point.priceChange,
            point.volume,
            point.fundingRateProxy,
            point.openInterestProxy,
            point.priceAcceleration,
            point.volumeSpike,
            point.priceGap,
            point.priceMomentum,
            point.volumeMomentum,
            point.rsi,
            point.bollingerPosition,
            point.marketRegime
        ]);

        const validationFeatures = validationData.map(point => [
            point.priceChange,
            point.volume,
            point.fundingRateProxy,
            point.openInterestProxy,
            point.priceAcceleration,
            point.volumeSpike,
            point.priceGap,
            point.priceMomentum,
            point.volumeMomentum,
            point.rsi,
            point.bollingerPosition,
            point.marketRegime
        ]);

        // Convert to TensorFlow tensors
        const X_train = tf.tensor2d(trainFeatures);
        const X_validation = tf.tensor2d(validationFeatures);

        console.log(`Prepared training data: ${trainFeatures.length} samples`);
        console.log(`Prepared validation data: ${validationFeatures.length} samples`);
        console.log(`Feature vector shape: [${trainFeatures[0].length}]`);
        console.log('First training sample:', trainFeatures[0]);
        console.log('First validation sample:', validationFeatures[0]);

        return {
            X_train,
            X_validation,
            trainData,
            validationData,
            trainFeatures,
            validationFeatures
        };
    }

    /**
     * Get data summary for display
     * @returns {Object} Data summary information
     */
    getDataSummary() {
        if (!this.processedData) {
            return null;
        }

        const totalPoints = this.processedData.length;
        const trainPoints = Math.floor(totalPoints * this.trainPercentage);
        const validationPoints = totalPoints - trainPoints;

        const dateRange = {
            start: this.processedData[0].datetime,
            end: this.processedData[totalPoints - 1].datetime
        };

        return {
            symbol: this.symbol,
            totalPoints,
            trainPoints,
            validationPoints,
            trainPercentage: this.trainPercentage,
            validationPercentage: this.validationPercentage,
            dateRange,
            lookbackDays: this.lookbackDays,
            interval: this.interval,
            features: [
                'priceChange', 'volume', 'fundingRateProxy', 'openInterestProxy', 
                'priceAcceleration', 'volumeSpike', 'priceGap', 'priceMomentum', 
                'volumeMomentum', 'rsi', 'bollingerPosition', 'marketRegime'
            ]
        };
    }

    /**
     * Analyze historical data for major market events
     * @param {Array} data - Processed market data
     * @returns {Object} Analysis of major events
     */
    analyzeHistoricalEvents(data) {
        const majorEvents = [];
        
        for (let i = 0; i < data.length; i++) {
            const point = data[i];
            
            // Detect major price drops (>5%)
            if (point.priceChange < -5) {
                majorEvents.push({
                    timestamp: point.timestamp,
                    datetime: point.datetime,
                    type: 'major_price_drop',
                    severity: point.priceChange < -10 ? 'critical' : 'warning',
                    priceChange: point.priceChange,
                    volumeChange: point.volumeChange,
                    volumeSpike: point.volumeSpike,
                    description: `Major price drop: ${point.priceChange.toFixed(2)}%`
                });
            }
            
            // Detect major volume spikes (>100%)
            if (point.volumeChange > 100) {
                majorEvents.push({
                    timestamp: point.timestamp,
                    datetime: point.datetime,
                    type: 'major_volume_spike',
                    severity: point.volumeChange > 200 ? 'critical' : 'warning',
                    priceChange: point.priceChange,
                    volumeChange: point.volumeChange,
                    volumeSpike: point.volumeSpike,
                    description: `Major volume spike: ${point.volumeChange.toFixed(2)}%`
                });
            }
            
            // Detect crash patterns (price drop + volume spike) - More sensitive for October 10th
            if (point.priceChange < -2 && point.volumeChange > 30) {
                majorEvents.push({
                    timestamp: point.timestamp,
                    datetime: point.datetime,
                    type: 'crash_pattern',
                    severity: 'critical',
                    priceChange: point.priceChange,
                    volumeChange: point.volumeChange,
                    volumeSpike: point.volumeSpike,
                    description: `Crash pattern: ${point.priceChange.toFixed(2)}% price drop + ${point.volumeChange.toFixed(2)}% volume spike`
                });
            }
            
            // Detect extreme market events (very sensitive for major crashes)
            if (point.priceChange < -1.5 && point.volumeChange > 20) {
                majorEvents.push({
                    timestamp: point.timestamp,
                    datetime: point.datetime,
                    type: 'extreme_event',
                    severity: point.priceChange < -3 ? 'critical' : 'warning',
                    priceChange: point.priceChange,
                    volumeChange: point.volumeChange,
                    volumeSpike: point.volumeSpike,
                    description: `Extreme event: ${point.priceChange.toFixed(2)}% price change + ${point.volumeChange.toFixed(2)}% volume change`
                });
            }
        }
        
        // Sort by severity and timestamp
        majorEvents.sort((a, b) => {
            if (a.severity !== b.severity) {
                return a.severity === 'critical' ? -1 : 1;
            }
            return new Date(b.timestamp) - new Date(a.timestamp);
        });
        
        // Enhanced debugging for October 10th events
        const october10Events = majorEvents.filter(e => 
            e.datetime.includes('2025-10-10') || 
            e.datetime.includes('2025-10-09') || 
            e.datetime.includes('2025-10-11')
        );
        
        // Log debugging information
        console.log(`Historical analysis: Found ${majorEvents.length} total events`);
        console.log(`October 10th events found: ${october10Events.length}`);
        if (october10Events.length === 0) {
            console.log('No October 10th events detected. Checking data range...');
            const dataRange = data.length > 0 ? {
                start: new Date(data[0].timestamp).toDateString(),
                end: new Date(data[data.length - 1].timestamp).toDateString()
            } : null;
            console.log('Data range:', dataRange);
        }
        
        return {
            totalEvents: majorEvents.length,
            criticalEvents: majorEvents.filter(e => e.severity === 'critical').length,
            warningEvents: majorEvents.filter(e => e.severity === 'warning').length,
            events: majorEvents.slice(0, 10), // Top 10 events
            october10Events: october10Events,
            dataRange: data.length > 0 ? {
                start: new Date(data[0].timestamp).toISOString(),
                end: new Date(data[data.length - 1].timestamp).toISOString()
            } : null
        };
    }

    /**
     * Update status message (for UI integration)
     * @param {string} message - Status message
     */
    updateStatus(message) {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = message;
        }
        console.log(message);
    }

    /**
     * Clean up tensors and free memory
     */
    dispose() {
        // Note: Tensors are created in prepareTrainingData and should be disposed by caller
        this.rawData = null;
        this.processedData = null;
        this.normalizedData = null;
        this.minMaxValues = {};
    }
}

export default BinanceDataLoader;

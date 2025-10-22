import BinanceDataLoader from './binance-data-loader.js';
import MLPAutoencoder, { EnsembleAutoencoder } from './autoencoder.js';

/**
 * AnomalyDetectionApp - Main application class for autoencoder-based anomaly detection
 * Handles UI interactions, data fetching, model training, and result visualization
 */
class AnomalyDetectionApp {
    constructor() {
        this.dataLoader = new BinanceDataLoader();
        this.autoencoder = new MLPAutoencoder(12); // Use single model for faster training
        this.currentSymbol = 'BTCUSDT';
        this.isTraining = false;
        this.isDetecting = false;
        this.trainingData = null;
        this.anomalyResults = null;
        
        // Chart instances
        this.lossChart = null;
        this.errorChart = null;
        this.severityChart = null;
        this.priceVolumeChart = null;
        
        this.initializeEventListeners();
        this.initializeUI();
    }

    /**
     * Initialize event listeners for UI controls
     */
    initializeEventListeners() {
        const symbolSelect = document.getElementById('symbolSelect');
        const fetchDataBtn = document.getElementById('fetchDataBtn');
        const trainBtn = document.getElementById('trainBtn');
        const detectBtn = document.getElementById('detectBtn');
        const saveModelBtn = document.getElementById('saveModelBtn');
        const loadModelBtn = document.getElementById('loadModelBtn');
        const resetBtn = document.getElementById('resetBtn');

        symbolSelect.addEventListener('change', (e) => this.handleSymbolChange(e));
        fetchDataBtn.addEventListener('click', () => this.fetchMarketData());
        trainBtn.addEventListener('click', () => this.trainModel());
        detectBtn.addEventListener('click', () => this.detectAnomalies());
        saveModelBtn.addEventListener('click', () => this.saveModel());
        loadModelBtn.addEventListener('click', () => this.loadModel());
        resetBtn.addEventListener('click', () => this.resetApplication());
    }

    /**
     * Initialize UI elements and set default states
     */
    initializeUI() {
        this.updateStatus('Select trading pair and click "Fetch Market Data" to begin');
        this.updateProgress(0);
        this.disableButtons(['trainBtn', 'detectBtn', 'saveModelBtn']);
        this.enableButtons(['fetchDataBtn', 'loadModelBtn', 'resetBtn']);
    }

    /**
     * Handle symbol selection change
     * @param {Event} event - Change event
     */
    handleSymbolChange(event) {
        this.currentSymbol = event.target.value;
        this.updateStatus(`Selected trading pair: ${this.currentSymbol}`);
    }

    /**
     * Fetch market data from Binance API with batch support
     */
    async fetchMarketData() {
        if (this.isTraining || this.isDetecting) return;
        
        try {
            // Get user preferences
            const useCustomRange = document.getElementById('useCustomRange').checked;
            const interval = document.getElementById('intervalSelect').value;
            const lookbackDays = parseInt(document.getElementById('lookbackDaysInput').value) || 60;
            
            let startDate = null;
            let endDate = null;
            
            if (useCustomRange) {
                const startDateInput = document.getElementById('startDateInput').value;
                const endDateInput = document.getElementById('endDateInput').value;
                
                if (startDateInput && endDateInput) {
                    startDate = new Date(startDateInput);
                    endDate = new Date(endDateInput);
                } else {
                    throw new Error('Please select both start and end dates for custom range');
                }
            } else {
                // Use default lookback period
                endDate = new Date();
                startDate = new Date(endDate.getTime() - (lookbackDays * 24 * 60 * 60 * 1000));
            }
            
            this.updateStatus(`Fetching market data from ${startDate.toISOString()} to ${endDate.toISOString()}...`);
            this.updateProgress(5);
            this.disableButtons(['fetchDataBtn']);

            // Use batch fetching with progress callback
            await this.dataLoader.fetchMarketDataBatched(
                this.currentSymbol, 
                startDate, 
                endDate, 
                interval,
                (currentBatch, totalBatches, message) => {
                    const progress = 5 + (currentBatch / totalBatches) * 25;
                    this.updateProgress(progress);
                    this.updateStatus(message);
                }
            );
            
            this.updateProgress(30);
            this.updateStatus('Processing and normalizing data...');
            this.dataLoader.normalizeData();
            this.updateProgress(60);
            
            this.trainingData = this.dataLoader.prepareTrainingData();
            this.updateProgress(80);
            
            const summary = this.dataLoader.getDataSummary();
            this.displayDataSummary(summary);
            this.displayDataSample();
            this.displayDetailedStatistics();
            this.createPriceVolumeChart();
            this.displayHistoricalEvents();
            this.updateProgress(100);
            
            this.enableButtons(['trainBtn', 'downloadDataBtn']);
            this.updateStatus('Data loaded successfully. Click "Train Autoencoder" to begin training.');
            
        } catch (error) {
            this.handleError('Data fetching error', error);
        }
    }

    /**
     * Display data sample table
     */
    displayDataSample() {
        const container = document.getElementById('dataSample');
        const content = document.getElementById('dataSampleContent');
        
        if (!container || !content || !this.dataLoader.processedData) return;

        const sampleData = this.dataLoader.processedData.slice(0, 10);
        
        content.innerHTML = `
            <table class="sample-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Open</th>
                        <th>High</th>
                        <th>Low</th>
                        <th>Close</th>
                        <th>Volume</th>
                        <th>Price Change %</th>
                        <th>Volume Change %</th>
                    </tr>
                </thead>
                <tbody>
                    ${sampleData.map(point => `
                        <tr>
                            <td>${new Date(point.timestamp).toLocaleString()}</td>
                            <td>${point.open.toFixed(2)}</td>
                            <td>${point.high.toFixed(2)}</td>
                            <td>${point.low.toFixed(2)}</td>
                            <td>${point.close.toFixed(2)}</td>
                            <td>${point.volume.toLocaleString()}</td>
                            <td style="color: ${point.priceChange >= 0 ? '#00ff00' : '#ff0000'}">${point.priceChange.toFixed(2)}%</td>
                            <td style="color: ${point.volumeChange >= 0 ? '#00ff00' : '#ff0000'}">${point.volumeChange.toFixed(2)}%</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;

        container.classList.remove('hidden');
    }

    /**
     * Display detailed statistics
     */
    displayDetailedStatistics() {
        if (!this.dataLoader.processedData) return;

        const data = this.dataLoader.processedData;
        
        // Calculate price statistics
        const prices = data.map(d => d.close);
        const priceStats = this.calculateStatistics(prices);
        
        // Calculate volume statistics
        const volumes = data.map(d => d.volume);
        const volumeStats = this.calculateStatistics(volumes);
        
        // Calculate change statistics
        const priceChanges = data.map(d => d.priceChange);
        const volumeChanges = data.map(d => d.volumeChange);
        const changeStats = {
            priceChange: this.calculateStatistics(priceChanges),
            volumeChange: this.calculateStatistics(volumeChanges)
        };
        
        // Calculate data quality metrics
        const qualityStats = this.calculateDataQuality(data);
        
        // Display price statistics
        document.getElementById('priceStats').innerHTML = this.formatStatistics(priceStats, 'USDT');
        
        // Display volume statistics
        document.getElementById('volumeStats').innerHTML = this.formatStatistics(volumeStats, 'units');
        
        // Display change statistics
        document.getElementById('changeStats').innerHTML = `
            <div class="stat-item">
                <span class="stat-label">Price Change Mean:</span>
                <span class="stat-value" style="color: ${changeStats.priceChange.mean >= 0 ? '#00ff00' : '#ff0000'}">${changeStats.priceChange.mean.toFixed(4)}%</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Price Change Std:</span>
                <span class="stat-value">${changeStats.priceChange.std.toFixed(4)}%</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Volume Change Mean:</span>
                <span class="stat-value" style="color: ${changeStats.volumeChange.mean >= 0 ? '#00ff00' : '#ff0000'}">${changeStats.volumeChange.mean.toFixed(4)}%</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Volume Change Std:</span>
                <span class="stat-value">${changeStats.volumeChange.std.toFixed(4)}%</span>
            </div>
        `;
        
        // Display quality statistics
        document.getElementById('qualityStats').innerHTML = `
            <div class="stat-item">
                <span class="stat-label">Missing Data:</span>
                <span class="stat-value" style="color: ${qualityStats.missingData === 0 ? '#00ff00' : '#ff0000'}">${qualityStats.missingData}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Zero Volume:</span>
                <span class="stat-value" style="color: ${qualityStats.zeroVolume === 0 ? '#00ff00' : '#ff0000'}">${qualityStats.zeroVolume}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Negative Prices:</span>
                <span class="stat-value" style="color: ${qualityStats.negativePrices === 0 ? '#00ff00' : '#ff0000'}">${qualityStats.negativePrices}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Data Completeness:</span>
                <span class="stat-value" style="color: ${qualityStats.completeness >= 95 ? '#00ff00' : qualityStats.completeness >= 90 ? '#ffff00' : '#ff0000'}">${qualityStats.completeness.toFixed(1)}%</span>
            </div>
        `;
        
        // Show corrupted data warnings if any
        if (qualityStats.missingData > 0 || qualityStats.zeroVolume > 0 || qualityStats.negativePrices > 0) {
            this.displayCorruptedData(qualityStats);
        }
        
        document.getElementById('statisticsContainer').classList.remove('hidden');
    }

    /**
     * Calculate basic statistics for an array
     * @param {Array} values - Array of numeric values
     * @returns {Object} Statistics object
     */
    calculateStatistics(values) {
        const validValues = values.filter(v => !isNaN(v) && isFinite(v));
        if (validValues.length === 0) return { mean: 0, std: 0, min: 0, max: 0 };
        
        const mean = validValues.reduce((sum, val) => sum + val, 0) / validValues.length;
        const variance = validValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / validValues.length;
        const std = Math.sqrt(variance);
        const min = Math.min(...validValues);
        const max = Math.max(...validValues);
        
        return { mean, std, min, max };
    }

    /**
     * Format statistics for display
     * @param {Object} stats - Statistics object
     * @param {string} unit - Unit suffix
     * @returns {string} HTML string
     */
    formatStatistics(stats, unit) {
        return `
            <div class="stat-item">
                <span class="stat-label">Mean:</span>
                <span class="stat-value">${stats.mean.toFixed(2)} ${unit}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Std Dev:</span>
                <span class="stat-value">${stats.std.toFixed(2)} ${unit}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Min:</span>
                <span class="stat-value">${stats.min.toFixed(2)} ${unit}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Max:</span>
                <span class="stat-value">${stats.max.toFixed(2)} ${unit}</span>
            </div>
        `;
    }

    /**
     * Calculate data quality metrics
     * @param {Array} data - Processed data array
     * @returns {Object} Quality metrics
     */
    calculateDataQuality(data) {
        let missingData = 0;
        let zeroVolume = 0;
        let negativePrices = 0;
        
        data.forEach(point => {
            if (!point.open || !point.close || !point.high || !point.low) missingData++;
            if (point.volume === 0) zeroVolume++;
            if (point.open <= 0 || point.close <= 0 || point.high <= 0 || point.low <= 0) negativePrices++;
        });
        
        const completeness = ((data.length - missingData) / data.length) * 100;
        
        return { missingData, zeroVolume, negativePrices, completeness };
    }

    /**
     * Display corrupted data warnings
     * @param {Object} qualityStats - Quality statistics
     */
    displayCorruptedData(qualityStats) {
        const container = document.getElementById('corruptedDataContainer');
        const content = document.getElementById('corruptedDataContent');
        
        if (!container || !content) return;
        
        const issues = [];
        if (qualityStats.missingData > 0) {
            issues.push(`${qualityStats.missingData} records with missing price data`);
        }
        if (qualityStats.zeroVolume > 0) {
            issues.push(`${qualityStats.zeroVolume} records with zero volume`);
        }
        if (qualityStats.negativePrices > 0) {
            issues.push(`${qualityStats.negativePrices} records with negative prices`);
        }
        
        content.innerHTML = `
            <p><strong>Issues Found:</strong></p>
            <ul style="margin-left: 20px; margin-top: 10px;">
                ${issues.map(issue => `<li style="color: #ff0000;">${issue}</li>`).join('')}
            </ul>
            <p style="margin-top: 10px; font-size: 12px; color: #cccccc;">
                These issues may affect anomaly detection accuracy. Consider filtering or cleaning the data.
            </p>
        `;
        
        container.classList.remove('hidden');
    }

    /**
     * Create price and volume timeline chart
     */
    createPriceVolumeChart() {
        if (!this.dataLoader.processedData) return;
        
        const ctx = document.getElementById('priceVolumeChart').getContext('2d');
        
        if (this.priceVolumeChart) {
            this.priceVolumeChart.destroy();
        }

        const data = this.dataLoader.processedData;
        const labels = data.map(d => new Date(d.timestamp).toLocaleDateString());
        const prices = data.map(d => d.close);
        const volumes = data.map(d => d.volume);
        
        // Normalize volume for dual-axis display
        const maxVolume = Math.max(...volumes);
        const normalizedVolumes = volumes.map(v => (v / maxVolume) * Math.max(...prices));

        this.priceVolumeChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Price (USDT)',
                        data: prices,
                        borderColor: '#00ff00',
                        backgroundColor: 'rgba(0, 255, 0, 0.1)',
                        yAxisID: 'y',
                        tension: 0.4
                    },
                    {
                        label: 'Volume (Normalized)',
                        data: normalizedVolumes,
                        borderColor: '#ffff00',
                        backgroundColor: 'rgba(255, 255, 0, 0.1)',
                        yAxisID: 'y1',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Date',
                            color: '#ffffff'
                        },
                        ticks: {
                            color: '#ffffff',
                            maxTicksLimit: 10
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Price (USDT)',
                            color: '#00ff00'
                        },
                        ticks: {
                            color: '#00ff00'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Volume (Normalized)',
                            color: '#ffff00'
                        },
                        ticks: {
                            color: '#ffff00'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: '#ffffff'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                if (context.datasetIndex === 0) {
                                    return `Price: ${context.parsed.y.toFixed(2)} USDT`;
                                } else {
                                    const originalVolume = volumes[context.dataIndex];
                                    return `Volume: ${originalVolume.toLocaleString()}`;
                                }
                            }
                        }
                    }
                }
            }
        });

        document.getElementById('priceVolumeChartContainer').classList.remove('hidden');
    }

    /**
     * Display data summary information
     * @param {Object} summary - Data summary object
     */
    displayDataSummary(summary) {
        const summaryElement = document.getElementById('dataSummary');
        const contentElement = document.getElementById('dataSummaryContent');
        
        if (summaryElement && contentElement) {
            contentElement.innerHTML = `
                <div class="metrics-grid">
                    <div class="metric-item">
                        <span class="metric-label">Trading Pair:</span>
                        <span class="metric-value">${summary.symbol}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Total Data Points:</span>
                        <span class="metric-value">${summary.totalPoints}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Training Samples:</span>
                        <span class="metric-value">${summary.trainPoints} (${(summary.trainPercentage * 100).toFixed(1)}%)</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Validation Samples:</span>
                        <span class="metric-value">${summary.validationPoints} (${(summary.validationPercentage * 100).toFixed(1)}%)</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Date Range:</span>
                        <span class="metric-value">${summary.dateRange.start.split('T')[0]} to ${summary.dateRange.end.split('T')[0]}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Interval:</span>
                        <span class="metric-value">${summary.interval}</span>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <strong>Features:</strong> ${summary.features.join(', ')}
                </div>
            `;
            summaryElement.classList.remove('hidden');
        }
    }

    /**
     * Train the autoencoder model
     */
    async trainModel() {
        if (!this.trainingData) {
            alert('Please fetch market data first');
            return;
        }

        if (this.isTraining) return;
        
        this.isTraining = true;
        this.disableButtons(['trainBtn', 'detectBtn', 'saveModelBtn']);
        
        try {
            const epochsInput = document.getElementById('epochsInput');
            const batchSizeInput = document.getElementById('batchSizeInput');
            
            // Use user input values (no forced overrides)
            const epochs = parseInt(epochsInput.value) || 30;
            const batchSize = parseInt(batchSizeInput.value) || 64;
            
            console.log(`Training with ${epochs} epochs and batch size ${batchSize} (user specified)`);
            
            const history = await this.autoencoder.train(
                this.trainingData.X_train,
                this.trainingData.X_validation,
                epochs,
                batchSize
            );
            
            this.createLossChart(history);
            this.enableButtons(['detectBtn', 'saveModelBtn']);
            this.updateStatus('Training completed successfully! Click "Detect Anomalies" to analyze the data.');
            
        } catch (error) {
            this.handleError('Training error', error);
        } finally {
            this.isTraining = false;
        }
    }

    /**
     * Create training loss chart
     * @param {Object} history - Training history
     */
    createLossChart(history) {
        const ctx = document.getElementById('lossChart').getContext('2d');
        
        if (this.lossChart) {
            this.lossChart.destroy();
        }

        const epochs = Array.from({ length: history.epoch.length }, (_, i) => i + 1);
        
        this.lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Training Loss',
                        data: history.history.loss,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.4
                    },
                    {
                        label: 'Validation Loss',
                        data: history.history.val_loss,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Loss'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true
                    }
                }
            }
        });

        document.getElementById('lossChartContainer').classList.remove('hidden');
    }

    /**
     * Detect anomalies in the data
     */
    async detectAnomalies() {
        if (!this.trainingData) {
            alert('Please fetch market data first');
            return;
        }

        if (!this.autoencoder.model) {
            alert('Please train the model first');
            return;
        }

        if (this.isDetecting) return;
        
        this.isDetecting = true;
        this.disableButtons(['detectBtn']);
        this.updateStatus('Detecting anomalies...');

        try {
            // Detect anomalies in validation data
            this.anomalyResults = this.autoencoder.detectAnomalies(
                this.trainingData.X_validation,
                this.trainingData.validationData
            );
            
            this.createErrorChart();
            this.createSeverityChart();
            this.displayAnomalyTable();
            this.displayFeatureContribution();
            
            this.updateStatus(`Anomaly detection completed. Found ${this.anomalyResults.anomalies.filter(a => a.severity !== 'normal').length} anomalies.`);
            
        } catch (error) {
            this.handleError('Anomaly detection error', error);
        } finally {
            this.isDetecting = false;
            this.enableButtons(['detectBtn']);
        }
    }

    /**
     * Create reconstruction error timeline chart
     */
    createErrorChart() {
        const ctx = document.getElementById('errorChart').getContext('2d');
        
        if (this.errorChart) {
            this.errorChart.destroy();
        }

        const anomalies = this.anomalyResults.anomalies;
        const labels = anomalies.map(a => a.metadata ? a.metadata.datetime.split('T')[0] : `Point ${a.index}`);
        const errors = anomalies.map(a => a.error);
        const colors = anomalies.map(a => {
            switch (a.severity) {
                case 'critical': return 'rgba(220, 53, 69, 0.8)';
                case 'warning': return 'rgba(255, 193, 7, 0.8)';
                default: return 'rgba(40, 167, 69, 0.8)';
            }
        });

        this.errorChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Reconstruction Error',
                    data: errors.map((error, index) => ({ x: index, y: error })),
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.8', '1')),
                    borderWidth: 1,
                    pointRadius: 4
                }, {
                    label: 'Anomaly Threshold',
                    data: errors.map((_, index) => ({ x: index, y: this.anomalyResults.threshold })),
                    type: 'line',
                    borderColor: 'rgba(108, 117, 125, 0.8)',
                    backgroundColor: 'rgba(108, 117, 125, 0.2)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Reconstruction Error'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time Points'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                if (context.datasetIndex === 0) {
                                    const anomaly = anomalies[context.dataIndex];
                                    return `Error: ${anomaly.error.toFixed(6)}, Severity: ${anomaly.severity}`;
                                }
                                return `Threshold: ${this.anomalyResults.threshold.toFixed(6)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Create anomaly severity distribution chart
     */
    createSeverityChart() {
        const ctx = document.getElementById('severityChart').getContext('2d');
        
        if (this.severityChart) {
            this.severityChart.destroy();
        }

        const severityLevels = this.anomalyResults.severityLevels;
        
        this.severityChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Normal', 'Warning', 'Critical'],
                datasets: [{
                    data: [severityLevels.normal, severityLevels.warning, severityLevels.critical],
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.8)',
                        'rgba(255, 193, 7, 0.8)',
                        'rgba(220, 53, 69, 0.8)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const total = severityLevels.normal + severityLevels.warning + severityLevels.critical;
                                const percentage = total > 0 ? ((context.raw / total) * 100).toFixed(1) : 0;
                                return `${context.label}: ${context.raw} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Display anomaly table
     */
    displayAnomalyTable() {
        const container = document.getElementById('anomalyTableContainer');
        const content = document.getElementById('anomalyTableContent');
        
        if (!container || !content) return;

        const topAnomalies = this.anomalyResults.anomalies
            .filter(a => a.severity !== 'normal')
            .slice(0, 15);

        if (topAnomalies.length === 0) {
            content.innerHTML = '<p style="padding: 20px; text-align: center; color: #6c757d;">No anomalies detected above threshold.</p>';
        } else {
            content.innerHTML = `
                <table class="anomaly-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Timestamp</th>
                            <th>Error Score</th>
                            <th>Severity</th>
                            <th>Importance</th>
                            <th>Major Event</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${topAnomalies.map((anomaly, index) => {
                            const importance = anomaly.error / anomaly.threshold;
                            const isMajorEvent = anomaly.isMajorEvent || false;
                            const timestamp = anomaly.metadata ? 
                                new Date(anomaly.metadata.datetime).toLocaleString() : 
                                `Point ${anomaly.index}`;
                            
                            return `
                                <tr class="${isMajorEvent ? 'major-event-row' : ''}">
                                    <td class="rank">#${index + 1}</td>
                                    <td class="timestamp">${timestamp}</td>
                                    <td class="error-score">${anomaly.error.toFixed(6)}</td>
                                    <td><span class="severity-badge severity-${anomaly.severity}">${anomaly.severity}</span></td>
                                    <td class="importance">${importance.toFixed(2)}x</td>
                                    <td class="major-event">${isMajorEvent ? '🚨 YES' : 'No'}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
                <div class="anomaly-summary">
                    <p><strong>Total Anomalies:</strong> ${topAnomalies.length} | 
                       <strong>Critical:</strong> ${topAnomalies.filter(a => a.severity === 'critical').length} | 
                       <strong>Warning:</strong> ${topAnomalies.filter(a => a.severity === 'warning').length} | 
                       <strong>Major Events:</strong> ${topAnomalies.filter(a => a.isMajorEvent).length}</p>
                </div>
            `;
        }

        container.classList.remove('hidden');
    }

    /**
     * Display feature contribution analysis
     */
    displayFeatureContribution() {
        const container = document.getElementById('featureContributionContainer');
        const content = document.getElementById('featureContributionContent');
        
        if (!container || !content) return;

        const topAnomaly = this.anomalyResults.anomalies.find(a => a.severity !== 'normal');
        
        if (!topAnomaly || !topAnomaly.featureContributions) {
            content.innerHTML = '<p style="text-align: center; color: #6c757d;">No feature contribution data available.</p>';
        } else {
            content.innerHTML = `
                <h4>Top Anomaly Feature Analysis</h4>
                <div class="feature-contribution">
                    ${topAnomaly.featureContributions.map(feature => `
                        <div class="feature-item">
                            <div class="feature-name">${feature.feature}</div>
                            <div class="feature-value ${feature.contribution > 0 ? 'positive' : 'negative'}">
                                ${feature.contribution.toFixed(4)}
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        container.classList.remove('hidden');
    }

    /**
     * Save model weights
     */
    async saveModel() {
        if (!this.autoencoder.model) {
            alert('No model to save. Please train a model first.');
            return;
        }

        try {
            await this.autoencoder.saveWeights();
            this.updateStatus('Model weights saved successfully!');
        } catch (error) {
            this.handleError('Failed to save model', error);
        }
    }

    /**
     * Load model weights
     */
    async loadModel() {
        try {
            await this.autoencoder.loadWeights();
            this.enableButtons(['detectBtn', 'saveModelBtn']);
            this.updateStatus('Model weights loaded successfully!');
        } catch (error) {
            this.handleError('Failed to load model', error);
        }
    }

    /**
     * Reset the entire application
     */
    resetApplication() {
        // Clean up existing resources
        this.dispose();
        
        // Reset UI to reasonable defaults (user can still change these)
        document.getElementById('symbolSelect').value = 'BTCUSDT';
        document.getElementById('epochsInput').value = '30';
        document.getElementById('batchSizeInput').value = '64';
        document.getElementById('lookbackDaysInput').value = '60';
        document.getElementById('intervalSelect').value = '1h';
        document.getElementById('useCustomRange').checked = false;
        
        // Hide all result sections
        document.getElementById('dataSummary').classList.add('hidden');
        document.getElementById('dataSample').classList.add('hidden');
        document.getElementById('statisticsContainer').classList.add('hidden');
        document.getElementById('corruptedDataContainer').classList.add('hidden');
        document.getElementById('priceVolumeChartContainer').classList.add('hidden');
        document.getElementById('lossChartContainer').classList.add('hidden');
        document.getElementById('anomalyTableContainer').classList.add('hidden');
        document.getElementById('featureContributionContainer').classList.add('hidden');
        
        // Reset state
        this.currentSymbol = 'BTCUSDT';
        this.isTraining = false;
        this.isDetecting = false;
        this.trainingData = null;
        this.anomalyResults = null;
        
        // Reinitialize
        this.dataLoader = new BinanceDataLoader();
        this.autoencoder = new MLPAutoencoder(12); // Use single model for faster training
        this.initializeUI();
        
        // Ensure buttons are in correct state after reset
        this.enableButtons(['fetchDataBtn', 'loadModelBtn', 'resetBtn']);
        this.disableButtons(['trainBtn', 'detectBtn', 'saveModelBtn', 'downloadDataBtn']);
        
        // Debug: Check button states
        console.log('Reset completed. Button states:');
        console.log('fetchDataBtn disabled:', document.getElementById('fetchDataBtn').disabled);
        console.log('trainBtn disabled:', document.getElementById('trainBtn').disabled);
        
        this.updateStatus('Application reset. Select trading pair and click "Fetch Market Data" to begin.');
    }

    /**
     * Update status message
     * @param {string} message - Status message
     */
    updateStatus(message) {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = message;
        }
    }

    /**
     * Update progress bar
     * @param {number} progress - Progress percentage (0-100)
     */
    updateProgress(progress) {
        const progressElement = document.getElementById('trainingProgress');
        if (progressElement) {
            progressElement.value = progress;
        }
    }

    /**
     * Enable specified buttons
     * @param {Array} buttonIds - Array of button IDs to enable
     */
    enableButtons(buttonIds) {
        buttonIds.forEach(id => {
            const button = document.getElementById(id);
            if (button) button.disabled = false;
        });
    }

    /**
     * Disable specified buttons
     * @param {Array} buttonIds - Array of button IDs to disable
     */
    disableButtons(buttonIds) {
        buttonIds.forEach(id => {
            const button = document.getElementById(id);
            if (button) button.disabled = true;
        });
    }

    /**
     * Handle errors with user-friendly messages
     * @param {string} context - Error context
     * @param {Error} error - Error object
     */
    handleError(context, error) {
        console.error(`${context}:`, error);
        this.updateStatus(`Error: ${error.message}`);
        this.updateProgress(0);
        
        // Re-enable relevant buttons
        if (this.trainingData) {
            this.enableButtons(['trainBtn']);
        } else {
            this.enableButtons(['fetchDataBtn']);
        }
    }

    /**
     * Clean up resources and free memory
     */
    dispose() {
        if (this.dataLoader) {
            this.dataLoader.dispose();
        }
        if (this.autoencoder) {
            this.autoencoder.dispose();
        }
        if (this.trainingData) {
            this.trainingData.X_train.dispose();
            this.trainingData.X_validation.dispose();
        }
        if (this.lossChart) {
            this.lossChart.destroy();
            this.lossChart = null;
        }
        if (this.errorChart) {
            this.errorChart.destroy();
            this.errorChart = null;
        }
        if (this.severityChart) {
            this.severityChart.destroy();
            this.severityChart = null;
        }
        if (this.priceVolumeChart) {
            this.priceVolumeChart.destroy();
            this.priceVolumeChart = null;
        }
    }

    /**
     * Display historical major events analysis
     */
    displayHistoricalEvents() {
        if (!this.dataLoader.processedData) return;
        
        const historicalAnalysis = this.dataLoader.analyzeHistoricalEvents(this.dataLoader.processedData);
        
        // Create or update historical events container
        let container = document.getElementById('historicalEvents');
        if (!container) {
            container = document.createElement('div');
            container.id = 'historicalEvents';
            container.className = 'historical-events';
            container.innerHTML = `
                <h4>Historical Major Events Analysis</h4>
                <div class="events-summary">
                    <div class="summary-item">
                        <span class="label">Total Events:</span>
                        <span class="value">${historicalAnalysis.totalEvents}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Critical Events:</span>
                        <span class="value critical">${historicalAnalysis.criticalEvents}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Warning Events:</span>
                        <span class="value warning">${historicalAnalysis.warningEvents}</span>
                    </div>
                </div>
                <div class="events-list">
                    <h5>Top Major Events:</h5>
                    <table class="major-events-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Date & Time</th>
                                <th>Event Type</th>
                                <th>Severity</th>
                                <th>Price Change</th>
                                <th>Volume Change</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${historicalAnalysis.events.map((event, index) => `
                                <tr class="event-row ${event.severity}">
                                    <td class="event-rank">#${index + 1}</td>
                                    <td class="event-time">${new Date(event.datetime).toLocaleString()}</td>
                                    <td class="event-type">${event.type.replace('_', ' ').toUpperCase()}</td>
                                    <td><span class="severity-badge severity-${event.severity}">${event.severity}</span></td>
                                    <td class="price-change ${event.priceChange < 0 ? 'negative' : 'positive'}">${event.priceChange.toFixed(2)}%</td>
                                    <td class="volume-change ${event.volumeChange > 0 ? 'positive' : 'negative'}">${event.volumeChange.toFixed(2)}%</td>
                                    <td class="event-description">${event.description}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                ${historicalAnalysis.october10Events.length > 0 ? `
                    <div class="october10-events">
                        <h5>October 10th Events Found:</h5>
                        <table class="october10-table">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Event Type</th>
                                    <th>Price Change</th>
                                    <th>Volume Change</th>
                                    <th>Description</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${historicalAnalysis.october10Events.map(event => `
                                    <tr class="october10-row ${event.severity}">
                                        <td class="event-time">${new Date(event.datetime).toLocaleString()}</td>
                                        <td class="event-type">${event.type.replace('_', ' ').toUpperCase()}</td>
                                        <td class="price-change ${event.priceChange < 0 ? 'negative' : 'positive'}">${event.priceChange.toFixed(2)}%</td>
                                        <td class="volume-change ${event.volumeChange > 0 ? 'positive' : 'negative'}">${event.volumeChange.toFixed(2)}%</td>
                                        <td class="event-description">${event.description}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                ` : `
                    <div class="october10-events">
                        <h5>October 10th Analysis:</h5>
                        <div class="no-events">No major events detected around October 10th in current data range.</div>
                    </div>
                `}
            `;
            
            // Insert after price volume chart
            const priceVolumeContainer = document.getElementById('priceVolumeChartContainer');
            if (priceVolumeContainer) {
                priceVolumeContainer.parentNode.insertBefore(container, priceVolumeContainer.nextSibling);
            }
        }
        
        container.classList.remove('hidden');
    }

    /**
     * Download raw data as JSON file
     */
    downloadRawData() {
        try {
            this.dataLoader.downloadRawData();
            this.updateStatus('Raw data downloaded successfully');
        } catch (error) {
            this.handleError('Download error', error);
        }
    }

    /**
     * Initialize UI controls and set default values
     */
    initializeUI() {
        // Set default date values
        const now = new Date();
        const defaultStart = new Date(now.getTime() - (60 * 24 * 60 * 60 * 1000)); // 60 days ago
        
        document.getElementById('endDateInput').value = now.toISOString().slice(0, 16);
        document.getElementById('startDateInput').value = defaultStart.toISOString().slice(0, 16);
        
        // Only add event listeners if they haven't been added before
        const useCustomRangeCheckbox = document.getElementById('useCustomRange');
        if (!useCustomRangeCheckbox.hasAttribute('data-listener-added')) {
            useCustomRangeCheckbox.addEventListener('change', (e) => {
                const startDateInput = document.getElementById('startDateInput');
                const endDateInput = document.getElementById('endDateInput');
                const lookbackDaysInput = document.getElementById('lookbackDaysInput');
                
                if (e.target.checked) {
                    startDateInput.disabled = false;
                    endDateInput.disabled = false;
                    lookbackDaysInput.disabled = true;
                } else {
                    startDateInput.disabled = true;
                    endDateInput.disabled = true;
                    lookbackDaysInput.disabled = false;
                }
            });
            useCustomRangeCheckbox.setAttribute('data-listener-added', 'true');
        }
        
        // Initialize custom range controls as disabled
        document.getElementById('startDateInput').disabled = true;
        document.getElementById('endDateInput').disabled = true;
        
        // Only add download button event listener if not already added
        const downloadBtn = document.getElementById('downloadDataBtn');
        if (!downloadBtn.hasAttribute('data-listener-added')) {
            downloadBtn.addEventListener('click', () => {
                this.downloadRawData();
            });
            downloadBtn.setAttribute('data-listener-added', 'true');
        }
        
        console.log('UI controls initialized with batch fetching support');
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AnomalyDetectionApp();
});

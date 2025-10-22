/**
 * MLP Autoencoder class for anomaly detection in cryptocurrency market data
 * Uses TensorFlow.js to build and train a multi-layer perceptron autoencoder
 */
class MLPAutoencoder {
    constructor(inputSize = 12) {
        this.model = null;
        this.inputSize = inputSize;
        this.history = null;
        this.threshold = null;
        this.trainingData = null;
        this.isTraining = false;
        this.isTrained = false;
        
        // Model architecture parameters
        this.encoderUnits = [16, 8, 4];
        this.latentSize = 2;
        this.decoderUnits = [4, 8, 16];
        this.dropoutRate = 0.2;
        
        // Training parameters
        this.learningRate = 0.001;
        this.epochs = 30; // Reduced for faster training
        this.batchSize = 64; // Larger batch size
        this.earlyStoppingPatience = 5; // Stop if no improvement for 5 epochs
    }

    /**
     * Build the autoencoder model architecture
     * @returns {tf.LayersModel} Compiled autoencoder model
     */
    buildModel() {
        // Encoder
        const encoder = tf.sequential({
            layers: [
                tf.layers.dense({
                    units: this.encoderUnits[0],
                    activation: 'relu',
                    inputShape: [this.inputSize]
                }),
                tf.layers.dropout({ rate: this.dropoutRate }),
                tf.layers.dense({
                    units: this.encoderUnits[1],
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: this.encoderUnits[2],
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: this.latentSize,
                    activation: 'relu',
                    name: 'latent'
                })
            ]
        });

        // Decoder
        const decoder = tf.sequential({
            layers: [
                tf.layers.dense({
                    units: this.decoderUnits[0],
                    activation: 'relu',
                    inputShape: [this.latentSize]
                }),
                tf.layers.dense({
                    units: this.decoderUnits[1],
                    activation: 'relu'
                }),
                tf.layers.dropout({ rate: this.dropoutRate }),
                tf.layers.dense({
                    units: this.decoderUnits[2],
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: this.inputSize,
                    activation: 'linear',
                    name: 'output'
                })
            ]
        });

        // Combine encoder and decoder
        const input = tf.input({ shape: [this.inputSize] });
        const encoded = encoder.apply(input);
        const decoded = decoder.apply(encoded);

        this.model = tf.model({ inputs: input, outputs: decoded });

        // Compile model
        this.model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });

        console.log('MLP Autoencoder model built successfully');
        console.log('Model summary:', this.model.summary());
        
        return this.model;
    }

    /**
     * Train the autoencoder model
     * @param {tf.Tensor} X_train - Training data tensor
     * @param {tf.Tensor} X_validation - Validation data tensor
     * @param {number} epochs - Number of training epochs
     * @param {number} batchSize - Batch size for training
     * @returns {Promise<tf.History>} Training history
     */
    async train(X_train, X_validation, epochs = null, batchSize = null) {
        if (!this.model) {
            this.buildModel();
        }

        this.epochs = epochs || this.epochs;
        this.batchSize = batchSize || this.batchSize;
        this.trainingData = X_train;
        
        console.log('Autoencoder training input shapes:');
        console.log('X_train shape:', X_train.shape);
        console.log('X_validation shape:', X_validation.shape);
        console.log('Expected input size:', this.inputSize);

        this.updateStatus('Starting autoencoder training...');

        try {
            let bestValLoss = Infinity;
            let patienceCounter = 0;
            
            this.history = await this.model.fit(X_train, X_train, {
                epochs: this.epochs,
                batchSize: this.batchSize,
                validationData: [X_validation, X_validation],
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        const progress = ((epoch + 1) / this.epochs) * 100;
                        const status = `Epoch ${epoch + 1}/${this.epochs} - Loss: ${logs.loss.toFixed(6)}, Val Loss: ${logs.val_loss.toFixed(6)}`;
                        
                        this.updateProgress(progress);
                        this.updateStatus(status);
                        
                        console.log(status);
                        tf.nextFrame(); // Prevent UI blocking
                        
                        // Early stopping logic
                        if (logs.val_loss < bestValLoss) {
                            bestValLoss = logs.val_loss;
                            patienceCounter = 0;
                        } else {
                            patienceCounter++;
                        }
                        
                        // Stop if no improvement for patience epochs
                        if (patienceCounter >= this.earlyStoppingPatience) {
                            console.log(`Early stopping at epoch ${epoch + 1} - no improvement for ${this.earlyStoppingPatience} epochs`);
                            this.model.stopTraining = true;
                        }
                    }
                }
            });

            // Calculate anomaly threshold from training data
            this.calculateThreshold(X_train);
            
            this.updateStatus('Training completed successfully!');
            this.updateProgress(100);
            this.isTrained = true;
            this.isTraining = false;
            
            return this.history;
            
        } catch (error) {
            throw new Error(`Training failed: ${error.message}`);
        }
    }

    /**
     * Calculate anomaly detection threshold from training data
     * @param {tf.Tensor} X_train - Training data tensor
     */
    calculateThreshold(X_train) {
        if (!this.model) {
            throw new Error('Model not trained');
        }

        // Get reconstruction errors for training data
        const reconstructions = this.model.predict(X_train);
        const errors = tf.mean(tf.square(tf.sub(X_train, reconstructions)), 1);
        const errorArray = errors.arraySync();
        
        // Calculate threshold: mean + 1.5 * std (even more sensitive for major events)
        const mean = errorArray.reduce((sum, val) => sum + val, 0) / errorArray.length;
        const variance = errorArray.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / errorArray.length;
        const std = Math.sqrt(variance);
        
        // Use a more sensitive threshold to catch major market events
        this.threshold = mean + 1.5 * std;
        
        // Also calculate percentiles for better threshold selection
        const sortedErrors = [...errorArray].sort((a, b) => a - b);
        const p95 = sortedErrors[Math.floor(sortedErrors.length * 0.95)];
        const p99 = sortedErrors[Math.floor(sortedErrors.length * 0.99)];
        
        // Use the more sensitive threshold between statistical and percentile methods
        this.threshold = Math.min(this.threshold, p95);
        
        console.log(`Anomaly threshold calculated: ${this.threshold.toFixed(6)}`);
        console.log(`Training data - Mean error: ${mean.toFixed(6)}, Std: ${std.toFixed(6)}`);
        
        // Clean up tensors
        reconstructions.dispose();
        errors.dispose();
    }

    /**
     * Detect anomalies in given data
     * @param {tf.Tensor} X - Data tensor to analyze
     * @param {Array} metadata - Optional metadata array (timestamps, etc.)
     * @returns {Object} Anomaly detection results
     */
    detectAnomalies(X, metadata = null) {
        if (!this.model || !this.threshold) {
            throw new Error('Model not trained or threshold not calculated');
        }

        // Get reconstructions and calculate errors
        const reconstructions = this.model.predict(X);
        const errors = tf.mean(tf.square(tf.sub(X, reconstructions)), 1);
        const errorArray = errors.arraySync();
        
        // Classify anomalies with more sensitive thresholds
        const anomalies = [];
        const severityLevels = { normal: 0, warning: 0, critical: 0 };
        
        // Calculate additional thresholds for better sensitivity to major events
        const warningThreshold = this.threshold;
        const criticalThreshold = this.threshold * 1.2; // Even more sensitive for major crashes
        
        for (let i = 0; i < errorArray.length; i++) {
            const error = errorArray[i];
            let severity = 'normal';
            
            if (error >= criticalThreshold) {
                severity = 'critical';
            } else if (error >= warningThreshold) {
                severity = 'warning';
            }
            
            severityLevels[severity]++;
            
            const anomaly = {
                index: i,
                error: error,
                severity: severity,
                threshold: this.threshold,
                warningThreshold: warningThreshold,
                criticalThreshold: criticalThreshold,
                metadata: metadata ? metadata[i] : null
            };
            
            // Calculate feature contributions
            if (metadata) {
                anomaly.featureContributions = this.calculateFeatureContributions(
                    X.slice(i, 1), 
                    reconstructions.slice(i, 1)
                );
                
                // Check for major market event patterns
                anomaly.isMajorEvent = this.detectMajorMarketEvent(anomaly.featureContributions, error);
            }
            
            anomalies.push(anomaly);
        }
        
        // Sort anomalies by error (highest first)
        anomalies.sort((a, b) => b.error - a.error);
        
        const results = {
            anomalies: anomalies,
            severityLevels: severityLevels,
            threshold: this.threshold,
            totalSamples: errorArray.length,
            anomalyRate: (severityLevels.warning + severityLevels.critical) / errorArray.length
        };
        
        // Clean up tensors
        reconstructions.dispose();
        errors.dispose();
        
        return results;
    }

    /**
     * Detect major market events based on feature patterns
     * @param {Array} featureContributions - Feature contribution analysis
     * @param {number} error - Reconstruction error
     * @returns {boolean} Whether this is a major market event
     */
    detectMajorMarketEvent(featureContributions, error) {
        if (!featureContributions) return false;
        
        // Find high-contributing features
        const highContributors = featureContributions.filter(f => f.contribution > 0.3);
        
        // Check for crash patterns: high price change + volume spike + acceleration
        const priceChangeContrib = featureContributions.find(f => f.feature === 'priceChange')?.contribution || 0;
        const volumeSpikeContrib = featureContributions.find(f => f.feature === 'volumeSpike')?.contribution || 0;
        const priceAccelContrib = featureContributions.find(f => f.feature === 'priceAcceleration')?.contribution || 0;
        const volumeContrib = featureContributions.find(f => f.feature === 'volume')?.contribution || 0;
        
        // Major event criteria:
        // 1. High reconstruction error
        // 2. Multiple high-contributing features
        // 3. Specific crash pattern (price + volume + acceleration)
        const hasHighError = error > this.threshold * 2; // Double threshold
        const hasMultipleHighContributors = highContributors.length >= 3;
        const hasCrashPattern = (priceChangeContrib > 0.4 && volumeSpikeContrib > 0.2) || 
                               (priceAccelContrib > 0.3 && volumeContrib > 0.2);
        
        return hasHighError && (hasMultipleHighContributors || hasCrashPattern);
    }

    /**
     * Calculate feature contribution to reconstruction error
     * @param {tf.Tensor} original - Original input tensor
     * @param {tf.Tensor} reconstruction - Reconstructed tensor
     * @returns {Array} Feature contributions
     */
    calculateFeatureContributions(original, reconstruction) {
        const originalArray = original.arraySync()[0];
        const reconstructionArray = reconstruction.arraySync()[0];
        
        const contributions = [];
        const featureNames = [
            'priceChange', 'volume', 'fundingRateProxy', 'openInterestProxy', 
            'priceAcceleration', 'volumeSpike', 'priceGap', 'priceMomentum', 
            'volumeMomentum', 'rsi', 'bollingerPosition', 'marketRegime'
        ];
        for (let i = 0; i < originalArray.length; i++) {
            const contribution = Math.abs(originalArray[i] - reconstructionArray[i]);
            contributions.push({
                feature: featureNames[i],
                contribution: contribution,
                originalValue: originalArray[i],
                reconstructedValue: reconstructionArray[i]
            });
        }
        
        return contributions;
    }

    /**
     * Get latent space representation of data
     * @param {tf.Tensor} X - Input data tensor
     * @returns {tf.Tensor} Latent space representation
     */
    encode(X) {
        if (!this.model) {
            throw new Error('Model not trained');
        }

        // Extract encoder part of the model
        const encoder = tf.model({
            inputs: this.model.input,
            outputs: this.model.getLayer('latent').output
        });

        return encoder.predict(X);
    }

    /**
     * Reconstruct data from latent representation
     * @param {tf.Tensor} latent - Latent space tensor
     * @returns {tf.Tensor} Reconstructed data
     */
    decode(latent) {
        if (!this.model) {
            throw new Error('Model not trained');
        }

        // Extract decoder part of the model
        const decoder = tf.model({
            inputs: this.model.getLayer('latent').output,
            outputs: this.model.output
        });

        return decoder.predict(latent);
    }

    /**
     * Save model weights to browser storage
     */
    async saveWeights() {
        if (!this.model) {
            throw new Error('No model to save');
        }

        try {
            // Save model weights
            await this.model.save('indexeddb://autoencoder-model');
            
            // Save additional parameters
            const modelData = {
                threshold: this.threshold,
                inputSize: this.inputSize,
                encoderUnits: this.encoderUnits,
                latentSize: this.latentSize,
                decoderUnits: this.decoderUnits,
                dropoutRate: this.dropoutRate,
                learningRate: this.learningRate,
                timestamp: Date.now()
            };
            
            localStorage.setItem('autoencoder-config', JSON.stringify(modelData));
            
            console.log('Model weights and configuration saved successfully');
            
        } catch (error) {
            throw new Error(`Failed to save model: ${error.message}`);
        }
    }

    /**
     * Load model weights from browser storage
     */
    async loadWeights() {
        try {
            // Load model weights
            this.model = await tf.loadLayersModel('indexeddb://autoencoder-model');
            
            // Load additional parameters
            const modelDataStr = localStorage.getItem('autoencoder-config');
            if (modelDataStr) {
                const modelData = JSON.parse(modelDataStr);
                this.threshold = modelData.threshold;
                this.inputSize = modelData.inputSize;
                this.encoderUnits = modelData.encoderUnits;
                this.latentSize = modelData.latentSize;
                this.decoderUnits = modelData.decoderUnits;
                this.dropoutRate = modelData.dropoutRate;
                this.learningRate = modelData.learningRate;
            }
            
            console.log('Model weights and configuration loaded successfully');
            
        } catch (error) {
            throw new Error(`Failed to load model: ${error.message}`);
        }
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
     * Update progress bar (for UI integration)
     * @param {number} progress - Progress percentage (0-100)
     */
    updateProgress(progress) {
        const progressElement = document.getElementById('trainingProgress');
        if (progressElement) {
            progressElement.value = progress;
        }
    }

    /**
     * Get model summary information
     * @returns {Object} Model information
     */
    getModelInfo() {
        return {
            inputSize: this.inputSize,
            encoderUnits: this.encoderUnits,
            latentSize: this.latentSize,
            decoderUnits: this.decoderUnits,
            dropoutRate: this.dropoutRate,
            learningRate: this.learningRate,
            threshold: this.threshold,
            isTrained: this.model !== null && this.threshold !== null
        };
    }

    /**
     * Clean up model and free memory
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        if (this.history) {
            this.history = null;
        }
        this.threshold = null;
        this.trainingData = null;
        this.isTraining = false;
        this.isTrained = false;
    }
}

/**
 * Ensemble Autoencoder for improved accuracy
 * Uses multiple autoencoders with different architectures and voting
 */
class EnsembleAutoencoder {
    constructor(inputSize = 12) {
        this.inputSize = inputSize;
        this.models = [];
        this.thresholds = [];
        this.isTrained = false;
        
        // Create ensemble of 3 different autoencoders
        this.models.push(new MLPAutoencoder(inputSize)); // Standard model
        this.models.push(new MLPAutoencoder(inputSize)); // Will be configured differently
        this.models.push(new MLPAutoencoder(inputSize)); // Will be configured differently
    }

    /**
     * Train all models in the ensemble with optimized settings
     */
    async train(X_train, X_validation, epochs = 50, batchSize = 64) {
        console.log('Training ensemble of 3 autoencoders with optimized settings...');
        
        // Train each model with different configurations
        for (let i = 0; i < this.models.length; i++) {
            console.log(`Training model ${i + 1}/3...`);
            
            // Configure different architectures
            if (i === 1) {
                // Model 2: More aggressive dropout
                this.models[i].dropoutRate = 0.3;
                this.models[i].learningRate = 0.0008;
            } else if (i === 2) {
                // Model 3: Different architecture
                this.models[i].encoderUnits = [20, 10, 5];
                this.models[i].latentSize = 3;
                this.models[i].decoderUnits = [5, 10, 20];
            }
            
            // Use fewer epochs for faster training
            const modelEpochs = Math.min(epochs, 30); // Cap at 30 epochs per model
            await this.models[i].train(X_train, X_validation, modelEpochs, batchSize);
        }
        
        this.isTrained = true;
        console.log('Ensemble training completed!');
    }

    /**
     * Detect anomalies using ensemble voting
     */
    detectAnomalies(X, metadata = null) {
        if (!this.isTrained) {
            throw new Error('Ensemble not trained');
        }

        const ensembleResults = [];
        const allAnomalies = [];
        
        // Get predictions from each model
        for (let i = 0; i < this.models.length; i++) {
            const result = this.models[i].detectAnomalies(X, metadata);
            ensembleResults.push(result);
            allAnomalies.push(result.anomalies);
        }

        // Ensemble voting logic
        const finalAnomalies = [];
        const numModels = this.models.length;
        
        for (let i = 0; i < allAnomalies[0].length; i++) {
            let normalVotes = 0;
            let warningVotes = 0;
            let criticalVotes = 0;
            
            // Count votes from each model
            for (let j = 0; j < numModels; j++) {
                const severity = allAnomalies[j][i].severity;
                if (severity === 'normal') normalVotes++;
                else if (severity === 'warning') warningVotes++;
                else if (severity === 'critical') criticalVotes++;
            }
            
            // Determine final severity based on majority vote
            let finalSeverity = 'normal';
            if (criticalVotes >= Math.ceil(numModels / 2)) {
                finalSeverity = 'critical';
            } else if (warningVotes >= Math.ceil(numModels / 2)) {
                finalSeverity = 'warning';
            } else if (warningVotes + criticalVotes >= Math.ceil(numModels / 2)) {
                finalSeverity = 'warning';
            }
            
            // Calculate ensemble confidence score
            const confidence = Math.max(normalVotes, warningVotes, criticalVotes) / numModels;
            
            // Average reconstruction errors
            const avgError = allAnomalies.reduce((sum, anomalies) => sum + anomalies[i].error, 0) / numModels;
            
            finalAnomalies.push({
                index: i,
                error: avgError,
                severity: finalSeverity,
                confidence: confidence,
                votes: { normal: normalVotes, warning: warningVotes, critical: criticalVotes },
                metadata: metadata ? metadata[i] : null,
                featureContributions: allAnomalies[0][i].featureContributions // Use first model's contributions
            });
        }
        
        // Calculate ensemble severity levels
        const severityLevels = { normal: 0, warning: 0, critical: 0 };
        finalAnomalies.forEach(anomaly => {
            severityLevels[anomaly.severity]++;
        });
        
        return {
            anomalies: finalAnomalies.sort((a, b) => b.error - a.error),
            severityLevels: severityLevels,
            threshold: ensembleResults[0].threshold, // Use first model's threshold
            totalSamples: finalAnomalies.length,
            anomalyRate: (severityLevels.warning + severityLevels.critical) / finalAnomalies.length,
            ensembleConfidence: finalAnomalies.reduce((sum, a) => sum + a.confidence, 0) / finalAnomalies.length
        };
    }

    /**
     * Save ensemble models
     */
    async saveWeights() {
        for (let i = 0; i < this.models.length; i++) {
            await this.models[i].saveWeights();
        }
    }

    /**
     * Load ensemble models
     */
    async loadWeights() {
        for (let i = 0; i < this.models.length; i++) {
            await this.models[i].loadWeights();
        }
    }

    /**
     * Clean up ensemble resources
     */
    dispose() {
        this.models.forEach(model => model.dispose());
        this.models = [];
        this.thresholds = [];
    }
}

export default MLPAutoencoder;
export { EnsembleAutoencoder };

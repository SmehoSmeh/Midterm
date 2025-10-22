/**
 * Feature Selection Configuration for Autoencoder Anomaly Detection
 * 
 * This file contains predefined feature sets for different use cases
 * and performance requirements. Modify the active feature set in your
 * autoencoder implementation to experiment with different combinations.
 */

/**
 * Current 12-feature set (default implementation)
 * Comprehensive but may have some redundancy
 */
export const CURRENT_FEATURES = [
    'priceChange',        // Close-to-close percentage change
    'volume',             // Raw trading volume (normalized)
    'fundingRateProxy',   // Simulated funding rate
    'openInterestProxy',  // Simulated open interest
    'priceAcceleration',  // Second derivative of price changes
    'volumeSpike',        // Volume spikes > 30%
    'priceGap',           // Gap between open and previous close
    'priceMomentum',      // 3-period price momentum
    'volumeMomentum',     // 3-period volume momentum
    'rsi',               // 14-period RSI
    'bollingerPosition', // Position within Bollinger Bands
    'marketRegime'       // Market trend detection
];

/**
 * Core feature set (8 features) - RECOMMENDED
 * Optimal balance of performance and complexity
 * Removes redundant and low-value features
 */
export const CORE_FEATURES = [
    'priceChange',        // Essential price movement
    'volume',             // Market activity baseline
    'priceAcceleration',  // Critical for crash detection
    'volumeSpike',        // Direct anomaly indicator
    'priceGap',           // Flash crash detection
    'rsi',               // Momentum oscillator
    'bollingerPosition', // Volatility context
    'marketRegime'       // Trend context
];

/**
 * Enhanced feature set (10 features)
 * Comprehensive analysis with moderate complexity
 * Keeps momentum features for better trend detection
 */
export const ENHANCED_FEATURES = [
    'priceChange',        // Price movement
    'volume',             // Volume
    'priceAcceleration',  // Price acceleration
    'volumeSpike',        // Volume spikes
    'priceGap',           // Price gaps
    'priceMomentum',      // Price momentum
    'volumeMomentum',    // Volume momentum
    'rsi',               // RSI
    'bollingerPosition', // Bollinger position
    'marketRegime'       // Market regime
];

/**
 * Minimal feature set (6 features)
 * Fastest training with essential features only
 * Good for quick experiments and resource-constrained environments
 */
export const MINIMAL_FEATURES = [
    'priceChange',        // Price movement
    'volume',             // Volume
    'priceAcceleration',  // Price acceleration
    'volumeSpike',        // Volume spikes
    'rsi',               // RSI
    'bollingerPosition'  // Bollinger position
];

/**
 * Alternative feature set (8 features)
 * Replaces simulated features with volatility-based features
 */
export const ALTERNATIVE_FEATURES = [
    'priceChange',        // Price change
    'volume',             // Volume
    'priceAcceleration',  // Price acceleration
    'volumeSpike',        // Volume spikes
    'priceGap',           // Price gaps
    'rsi',               // RSI
    'bollingerPosition', // Bollinger position
    'volatility'          // Intraday volatility
];

/**
 * Feature importance rankings based on anomaly detection effectiveness
 */
export const FEATURE_IMPORTANCE = {
    // High importance (⭐⭐⭐) - Critical for anomaly detection
    'priceChange': 3,
    'priceAcceleration': 3,
    'volumeSpike': 3,
    'priceGap': 3,
    
    // Medium importance (⭐⭐) - Important for context
    'volume': 2,
    'rsi': 2,
    'bollingerPosition': 2,
    'marketRegime': 2,
    
    // Lower importance (⭐) - Redundant or limited value
    'priceMomentum': 1,
    'volumeMomentum': 1,
    'fundingRateProxy': 1,
    'openInterestProxy': 1
};

/**
 * Architecture configurations for different feature counts
 */
export const ARCHITECTURE_CONFIGS = {
    6: {
        inputSize: 6,
        encoderUnits: [10, 5, 2],
        latentSize: 2,
        decoderUnits: [2, 5, 10],
        dropoutRate: 0.2
    },
    8: {
        inputSize: 8,
        encoderUnits: [12, 6, 3],
        latentSize: 2,
        decoderUnits: [3, 6, 12],
        dropoutRate: 0.2
    },
    10: {
        inputSize: 10,
        encoderUnits: [14, 7, 3],
        latentSize: 2,
        decoderUnits: [3, 7, 14],
        dropoutRate: 0.2
    },
    12: {
        inputSize: 12,
        encoderUnits: [16, 8, 4],
        latentSize: 2,
        decoderUnits: [4, 8, 16],
        dropoutRate: 0.2
    }
};

/**
 * Feature descriptions and formulas
 */
export const FEATURE_DESCRIPTIONS = {
    priceChange: {
        description: 'Close-to-close percentage change',
        formula: '((close - previous_close) / previous_close) * 100',
        purpose: 'Captures basic price movements and trends',
        importance: 'High - Fundamental price indicator'
    },
    volume: {
        description: 'Raw trading volume (MinMax normalized)',
        formula: 'MinMax(volume)',
        purpose: 'Market activity and liquidity indicator',
        importance: 'Medium - Activity baseline'
    },
    priceAcceleration: {
        description: 'Second derivative of price changes',
        formula: 'priceChange - previous_priceChange',
        purpose: 'Detects sudden price accelerations (crash patterns)',
        importance: 'High - Critical for crash detection'
    },
    volumeSpike: {
        description: 'Volume spikes above 30% threshold',
        formula: 'max(0, volumeChange - 30)',
        purpose: 'Detects unusual volume surges',
        importance: 'High - Direct anomaly indicator'
    },
    priceGap: {
        description: 'Gap between opening price and previous close',
        formula: '|open - previous_close| / previous_close * 100',
        purpose: 'Identifies overnight gaps and flash crashes',
        importance: 'High - Flash crash detection'
    },
    priceMomentum: {
        description: '3-period price momentum',
        formula: '(close - close_3_periods_ago) / close_3_periods_ago * 100',
        purpose: 'Captures medium-term price trends',
        importance: 'Low - Redundant with priceAcceleration'
    },
    volumeMomentum: {
        description: '3-period volume momentum',
        formula: '(volume - volume_3_periods_ago) / volume_3_periods_ago * 100',
        purpose: 'Captures volume trend changes',
        importance: 'Low - Redundant with volumeSpike'
    },
    rsi: {
        description: '14-period Relative Strength Index',
        formula: 'RSI(14) normalized to 0-1',
        purpose: 'Momentum oscillator for overbought/oversold conditions',
        importance: 'Medium - Momentum context'
    },
    bollingerPosition: {
        description: 'Position within Bollinger Bands',
        formula: '(close - sma) / (upper_band - lower_band)',
        purpose: 'Volatility and mean reversion indicator',
        importance: 'Medium - Volatility context'
    },
    marketRegime: {
        description: 'Market trend detection',
        formula: 'Trend strength vs volatility analysis',
        purpose: 'Contextual market state classification',
        importance: 'Medium - Trend context'
    },
    fundingRateProxy: {
        description: 'Simulated funding rate for spot markets',
        formula: '(momentumFactor * 0.5 + volumeFactor * 0.3 + volatilityFactor * 0.2) * 0.01',
        purpose: 'Approximates futures funding rate behavior',
        importance: 'Low - Limited value for spot markets'
    },
    openInterestProxy: {
        description: 'Simulated open interest',
        formula: '(volumeFactor * 0.4 + tradeIntensity * 0.3 + priceMomentum * 0.3)',
        purpose: 'Estimates market positioning',
        importance: 'Low - Simulated feature'
    }
};

/**
 * Usage example:
 * 
 * // In your autoencoder class constructor:
 * import { CORE_FEATURES, ARCHITECTURE_CONFIGS } from './feature-selection-config.js';
 * 
 * constructor() {
 *     const config = ARCHITECTURE_CONFIGS[CORE_FEATURES.length];
 *     this.inputSize = config.inputSize;
 *     this.encoderUnits = config.encoderUnits;
 *     this.latentSize = config.latentSize;
 *     this.decoderUnits = config.decoderUnits;
 *     this.dropoutRate = config.dropoutRate;
 * }
 * 
 * // In your data loader:
 * const selectedFeatures = CORE_FEATURES; // or any other feature set
 * const trainFeatures = trainData.map(point => 
 *     selectedFeatures.map(feature => point[feature])
 * );
 */

/**
 * Feature selection testing utilities
 */
export const TESTING_CONFIGS = {
    ablation: {
        description: 'Test removing one feature at a time',
        method: 'Train models with n-1 features, compare performance'
    },
    forward: {
        description: 'Add features one by one starting with most important',
        method: 'Start with top feature, add next most important'
    },
    backward: {
        description: 'Remove features one by one starting with least important',
        method: 'Start with all features, remove least important'
    },
    random: {
        description: 'Test random feature combinations',
        method: 'Generate random subsets, measure performance'
    }
};

export default {
    CURRENT_FEATURES,
    CORE_FEATURES,
    ENHANCED_FEATURES,
    MINIMAL_FEATURES,
    ALTERNATIVE_FEATURES,
    FEATURE_IMPORTANCE,
    ARCHITECTURE_CONFIGS,
    FEATURE_DESCRIPTIONS,
    TESTING_CONFIGS
};

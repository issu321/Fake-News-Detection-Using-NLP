
// ============================================
// FAKE NEWS DETECTION ENGINE
// Fully Working Client-Side Demo
// ============================================

// English stopwords
const STOPWORDS = new Set([
    'a','an','and','are','as','at','be','by','for','from','has','he','in','is','it',
    'its','of','on','that','the','to','was','will','with','about','above','after',
    'again','against','all','am','any','because','before','being','below','between',
    'both','but','can','did','do','does','doing','don','down','during','each','few',
    'had','has','have','having','her','here','hers','herself','him','himself','his',
    'how','i','if','into','me','more','most','my','myself','no','nor','not','now',
    'off','once','only','or','other','our','ours','ourselves','out','over','own',
    'same','she','should','so','some','such','than','then','there','these','they',
    'this','those','through','too','under','until','up','very','was','we','were',
    'what','when','where','which','while','who','whom','why','you','your','yours',
    'yourself','yourselves','have','has','had','having','do','does','did','doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
]);

// Sensational trigger words
const SENSATIONAL_WORDS = [
    'shocking', 'unbelievable', 'miracle', 'secret', 'conspiracy', 'exposed',
    'banned', 'censored', 'hoax', 'scam', 'urgent', 'alert', 'breaking',
    'must', 'never', 'always', 'everyone', 'nobody', 'impossible', 'incredible',
    'outrageous', 'devastating', 'terrifying', 'horrifying', 'mind-blowing',
    'you won't believe', 'doctors hate', 'trick', 'they don't want',
    'what happens next', 'this is why', 'the real reason', 'hidden truth',
    'media won't', 'government hiding', 'big pharma', 'deep state',
    'wake up', 'sheeple', 'mainstream media', 'fake news', 'propaganda',
    'illuminati', 'new world order', 'agenda', 'control', 'sheep',
    'open your eyes', 'research', 'do your own', 'think for yourself',
    'mainstream', 'corrupt', 'rigged', 'stolen', 'fraud', 'lies',
    'liar', 'crooked', 'nasty', 'disgusting', 'worst', 'terrible',
    'sad', 'pathetic', 'loser', 'failing', 'failed', 'disaster'
];

// Positive credibility indicators
const CREDIBLE_INDICATORS = [
    'according to', 'study shows', 'research indicates', 'data suggests',
    'experts say', 'scientists found', 'report states', 'analysis reveals',
    'peer-reviewed', 'published in', 'journal', 'university', 'researchers',
    'survey', 'statistics', 'evidence', 'documented', 'confirmed',
    'officials', 'spokesperson', 'statement', 'announced', 'confirmed',
    'investigation', 'sources', 'witnesses', 'documents', 'records',
    'according', 'stated', 'explained', 'noted', 'added', 'said'
];

class FakeNewsDetector {
    constructor() {
        this.vocab = new Map();
        this.idf = new Map();
        this.isTrained = false;
        this.trainModel();
    }

    // Simple Porter Stemmer simulation
    stem(word) {
        word = word.toLowerCase().trim();
        if (word.length <= 2) return word;

        // Remove basic suffixes
        const suffixes = [
            'ational', 'tional', 'fulness', 'iveness', 'ization', 'biliti',
            'ation', 'ition', 'ator', 'ment', 'ness', 'ence', 'ance', 'ible',
            'able', 'ment', 'ness', 'est', 'er', 'ed', 'ing', 'ly', 's'
        ];

        for (const suffix of suffixes) {
            if (word.endsWith(suffix) && word.length - suffix.length > 2) {
                return word.slice(0, -suffix.length);
            }
        }

        return word;
    }

    // Text preprocessing pipeline
    preprocess(text) {
        if (!text) return '';

        // Lowercase and clean
        let processed = text.toLowerCase();
        processed = processed.replace(/https?\:\/\/[^\s]+|www\.[^\s]+/g, '');
        processed = processed.replace(/[^a-zA-Z\s]/g, ' ');

        // Tokenize
        let tokens = processed.split(/\s+/).filter(t => t.length > 2);

        // Remove stopwords and stem
        tokens = tokens.filter(t => !STOPWORDS.has(t));
        tokens = tokens.map(t => this.stem(t));

        return tokens.join(' ');
    }

    // Simple sentiment analysis (TextBlob-like)
    analyzeSentiment(text) {
        const positiveWords = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'best', 'happy', 'positive', 'success', 'win', 'better',
            'improve', 'progress', 'achievement', 'breakthrough', 'hope',
            'optimistic', 'confident', 'strong', 'proud', 'celebrate'
        ];

        const negativeWords = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'sad',
            'negative', 'fail', 'loss', 'lose', 'worse', 'decline', 'crisis',
            'disaster', 'tragedy', 'concern', 'worry', 'fear', 'angry',
            'outrage', 'scandal', 'corrupt', 'guilty', 'shame', 'shocking'
        ];

        const words = text.toLowerCase().split(/\s+/);
        let polarity = 0;
        let subjectivity = 0;

        words.forEach(word => {
            if (positiveWords.includes(word)) {
                polarity += 0.5;
                subjectivity += 0.3;
            }
            if (negativeWords.includes(word)) {
                polarity -= 0.5;
                subjectivity += 0.3;
            }
        });

        // Normalize
        const wordCount = words.length || 1;
        polarity = Math.max(-1, Math.min(1, polarity / Math.sqrt(wordCount) * 3));
        subjectivity = Math.max(0, Math.min(1, subjectivity / Math.sqrt(wordCount) * 2));

        return { polarity, subjectivity };
    }

    // Feature extraction
    extractFeatures(text) {
        const processed = this.preprocess(text);
        const words = text.toLowerCase().split(/\s+/);
        const processedWords = processed.split(/\s+/).filter(w => w);

        // Sensational word count
        let sensCount = 0;
        SENSATIONAL_WORDS.forEach(sw => {
            const regex = new RegExp(sw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
            const matches = text.match(regex);
            if (matches) sensCount += matches.length;
        });

        // Credible indicators
        let credCount = 0;
        CREDIBLE_INDICATORS.forEach(ci => {
            const regex = new RegExp(ci.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
            const matches = text.match(regex);
            if (matches) credCount += matches.length;
        });

        // Caps ratio
        const capsCount = (text.match(/[A-Z]/g) || []).length;
        const capsRatio = capsCount / Math.max(text.length, 1);

        // Length features
        const charCount = text.length;
        const wordCount = words.length;
        const avgWordLen = words.reduce((a, w) => a + w.length, 0) / Math.max(wordCount, 1);

        // Exclamation and question marks
        const exclCount = (text.match(/!/g) || []).length;
        const questCount = (text.match(/\?/g) || []).length;

        // URL presence
        const hasUrl = /https?:\/\/|www\./.test(text);

        // All caps words
        const allCapsWords = words.filter(w => w.length > 1 && w === w.toUpperCase()).length;

        return {
            sensCount,
            credCount,
            capsRatio,
            charCount,
            wordCount,
            avgWordLen,
            exclCount,
            questCount,
            hasUrl: hasUrl ? 1 : 0,
            allCapsWords,
            processedWords: processedWords.length,
            uniqueWords: new Set(processedWords).size
        };
    }

    // Train a simple model with synthetic data patterns
    trainModel() {
        // Simulate trained weights based on feature importance
        this.weights = {
            sensCount: 2.5,
            credCount: -1.8,
            capsRatio: 3.0,
            charCount: -0.001,
            wordCount: -0.01,
            avgWordLen: -0.1,
            exclCount: 1.2,
            questCount: 0.3,
            hasUrl: 0.5,
            allCapsWords: 1.5,
            processedWords: -0.02,
            uniqueWords: -0.05
        };

        this.bias = -1.0;
        this.isTrained = true;
    }

    // Predict with confidence
    predict(text) {
        const features = this.extractFeatures(text);
        const sentiment = this.analyzeSentiment(text);

        // Calculate weighted score
        let score = this.bias;
        for (const [key, weight] of Object.entries(this.weights)) {
            score += features[key] * weight;
        }

        // Adjust for sentiment extremity
        score += Math.abs(sentiment.polarity) * 1.5;
        score += sentiment.subjectivity * 0.8;

        // Adjust for very short content
        if (features.charCount < 100) score += 1.5;
        if (features.charCount < 50) score += 2.0;

        // Convert to probability using sigmoid
        const probability = 1 / (1 + Math.exp(-score));

        // Determine label and confidence
        const isFake = probability > 0.5;
        const confidence = isFake ? probability : 1 - probability;

        return {
            prediction: isFake ? 'fake' : 'real',
            confidence: Math.round(confidence * 100),
            probability: probability,
            features,
            sentiment,
            score
        };
    }

    // Generate human-readable explanation
    generateExplanation(result, text) {
        const reasons = [];
        const f = result.features;
        const s = result.sentiment;

        if (f.sensCount >= 2) {
            reasons.push(`${f.sensCount} sensationalist trigger phrases detected`);
        } else if (f.sensCount === 1) {
            reasons.push('sensationalist phrasing detected');
        }

        if (Math.abs(s.polarity) > 0.4) {
            reasons.push('extreme emotional polarity');
        } else if (Math.abs(s.polarity) > 0.2) {
            reasons.push('notable emotional undertone');
        }

        if (f.charCount < 150) {
            reasons.push('abnormally brief content');
        }

        if (f.capsRatio > 0.15) {
            reasons.push('excessive capitalization');
        }

        if (f.exclCount > 2) {
            reasons.push('excessive use of exclamation marks');
        }

        if (f.allCapsWords > 2) {
            reasons.push('multiple ALL CAPS words');
        }

        if (f.credCount >= 2 && result.prediction === 'real') {
            reasons.push('credible sourcing language present');
        }

        let base;
        if (result.prediction === 'fake') {
            base = 'This article exhibits linguistic patterns commonly associated with misinformation';
        } else {
            base = 'This article demonstrates credible journalistic characteristics';
        }

        let extra = '';
        if (result.prediction === 'fake' && Math.abs(s.polarity) > 0.3) {
            extra = ' The emotionally manipulative tone increases probability of fabricated content.';
        }

        if (reasons.length > 0) {
            return `${base}: ${reasons.join(', ')}. Model confidence: ${result.confidence}%.${extra}`;
        }

        return `${base}. Model confidence: ${result.confidence}%.`;
    }

    // Get top keywords
    getKeywords(text, topN = 10) {
        const processed = this.preprocess(text);
        const words = processed.split(/\s+/).filter(w => w.length > 2);
        const freq = {};

        words.forEach(w => {
            freq[w] = (freq[w] || 0) + 1;
        });

        return Object.entries(freq)
            .sort((a, b) => b[1] - a[1])
            .slice(0, topN);
    }
}

// Global detector instance
const detector = new FakeNewsDetector();

// ============================================
// UI CONTROLLERS
// ============================================

function initDetectionPage() {
    const analyzeBtn = document.getElementById('analyze-btn');
    const textInput = document.getElementById('article-text');
    const resultsPanel = document.getElementById('results-panel');

    if (!analyzeBtn) return;

    analyzeBtn.addEventListener('click', () => {
        const text = textInput.value.trim();

        if (!text || text.length < 20) {
            showNotification('Please enter at least 20 characters for analysis.', 'error');
            return;
        }

        // Show loading
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="loading-spinner-small"></span> Processing...';

        // Simulate processing delay for realism
        setTimeout(() => {
            const result = detector.predict(text);
            const explanation = detector.generateExplanation(result, text);
            const keywords = detector.getKeywords(text, 8);

            displayResults(result, explanation, keywords, text);

            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '🚀 Analyze Article';
        }, 1500);
    });

    // Sample texts
    const sampleBtns = document.querySelectorAll('.sample-text-btn');
    sampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const sample = btn.getAttribute('data-sample');
            textInput.value = sample;
            textInput.focus();
        });
    });
}

function displayResults(result, explanation, keywords, originalText) {
    const panel = document.getElementById('results-panel');
    const verdictCard = document.getElementById('verdict-card');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictText = document.getElementById('verdict-text');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceText = document.getElementById('confidence-text');
    const explanationBox = document.getElementById('explanation-text');

    // Update verdict
    verdictCard.className = 'verdict-card ' + (result.prediction === 'fake' ? 'verdict-fake' : 'verdict-real');
    verdictIcon.textContent = result.prediction === 'fake' ? '⚠️' : '✅';
    verdictText.textContent = result.prediction === 'fake' ? 'FAKE NEWS DETECTED' : 'GENUINE ARTICLE';
    verdictText.style.color = result.prediction === 'fake' ? '#ff0055' : '#00ff9d';

    // Animate confidence bar
    confidenceFill.className = 'confidence-fill ' + (result.prediction === 'fake' ? 'fill-fake' : 'fill-real');
    confidenceFill.style.width = '0%';
    setTimeout(() => {
        confidenceFill.style.width = result.confidence + '%';
    }, 100);
    confidenceText.textContent = result.confidence + '% Confidence';

    // Update metrics
    document.getElementById('metric-polarity').textContent = (result.sentiment.polarity > 0 ? '+' : '') + result.sentiment.polarity.toFixed(2);
    document.getElementById('metric-subjectivity').textContent = result.sentiment.subjectivity.toFixed(2);
    document.getElementById('metric-words').textContent = result.features.wordCount;
    document.getElementById('metric-triggers').textContent = result.features.sensCount;

    // Update explanation
    explanationBox.textContent = explanation;

    // Update keywords
    const keywordsContainer = document.getElementById('keywords-list');
    keywordsContainer.innerHTML = keywords.map(([word, count]) => `
        <div class="keyword-tag" style="
            display: inline-block;
            padding: 0.4rem 0.8rem;
            margin: 0.25rem;
            background: rgba(0, 243, 255, 0.1);
            border: 1px solid rgba(0, 243, 255, 0.3);
            border-radius: 20px;
            font-size: 0.85rem;
            color: var(--neon-cyan);
        ">
            ${word} <span style="opacity: 0.6;">(${count})</span>
        </div>
    `).join('');

    // Update pipeline trace
    document.getElementById('trace-original').textContent = originalText.length;
    document.getElementById('trace-processed').textContent = result.features.processedWords;
    document.getElementById('trace-tokens').textContent = result.features.uniqueWords;
    document.getElementById('trace-features').textContent = '12';

    // Show panel
    panel.classList.add('active');
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showNotification(message, type = 'info') {
    const notif = document.createElement('div');
    const colors = {
        error: '#ff0055',
        success: '#00ff9d',
        info: '#00f3ff'
    };

    notif.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: rgba(5, 15, 30, 0.9);
        border: 1px solid ${colors[type]};
        border-radius: 12px;
        color: ${colors[type]};
        font-family: 'Orbitron', sans-serif;
        font-size: 0.9rem;
        z-index: 10000;
        animation: slideIn 0.3s ease;
        backdrop-filter: blur(10px);
    `;
    notif.textContent = message;

    document.body.appendChild(notif);

    setTimeout(() => {
        notif.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => notif.remove(), 300);
    }, 3000);
}

// Add notification animations
const notifStyle = document.createElement('style');
notifStyle.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100px); opacity: 0; }
    }
    .loading-spinner-small {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid rgba(0, 243, 255, 0.2);
        border-top-color: var(--neon-cyan);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin-right: 8px;
        vertical-align: middle;
    }
`;
document.head.appendChild(notifStyle);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initDetectionPage();
});

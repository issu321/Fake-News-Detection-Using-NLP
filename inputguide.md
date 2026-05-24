# 📋 Input Guide — Fake News Detection Using NLP

## Article Input Examples

### Example 1: Fake News Pattern
**Title:** BREAKING: Secret Government Plot Exposed  
**Text:** Shocking documents reveal that the government has been hiding a secret plan to control the population using mind-altering chemicals in the water supply. This unbelievable conspiracy has been kept hidden for decades...

**Expected Output:** `FAKE` — High confidence due to sensationalist language and extreme emotional polarity.

### Example 2: Real News Pattern
**Title:** Federal Reserve Announces Interest Rate Decision  
**Text:** The Federal Reserve concluded its two-day policy meeting on Wednesday, voting unanimously to maintain the federal funds rate at its current target range of 5.25% to 5.50%...

**Expected Output:** `GENUINE` — Formal tone, factual structure, low sensationalism.

---

## CSV Upload Format

Your CSV file **must contain** a `text` column. An optional `title` column is supported.

```csv
title,text
"Federal Reserve Update","The Federal Reserve concluded its regular policy meeting by voting to maintain the benchmark interest rate..."
"Miracle Cure Found","Underground researchers uncovered a miraculous plant extract that eliminates chronic disease..."
```

After upload, the system will:
1. Preprocess every row
2. Run TF-IDF vectorization
3. Predict labels with confidence scores
4. Compute sentiment polarity
5. Allow you to download the full results

---

## Dashboard Usage

| Section | What You See |
|---------|--------------|
| **Label Distribution** | Pie chart of fake vs real articles in the dataset |
| **Model Performance** | Side-by-side accuracy and F1-score comparison |
| **Fake Keywords** | Most frequent stemmed terms in fake articles |
| **Real Keywords** | Most frequent stemmed terms in genuine articles |

---

## Analytics Explanation

| Metric | Meaning |
|--------|---------|
| **Confidence** | Model probability for the predicted class (0–100%) |
| **Sentiment Polarity** | -1 (negative) to +1 (positive); extreme values often correlate with fake news |
| **Subjectivity** | 0 (objective) to 1 (subjective); high values suggest opinion-driven content |
| **TF-IDF Score** | Importance of a term within the article relative to the corpus |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| App won't start | Ensure Python 3.11+ is installed and the virtual environment is activated |
| NLTK errors | Run `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the activated venv |
| Model not found | The app auto-trains on first run. Ensure `news_dataset.csv` is in the project root |
| CSS not loading | Verify `assets/styles.css` exists and the app is launched from the project root |
| Batch upload fails | Confirm your CSV has a column named exactly `text` (case-sensitive) |

---

## Tips for Best Results

- Use articles with **at least 50 words** for reliable TF-IDF feature extraction
- Avoid bullet-point lists; the model is trained on paragraph-style news text
- For batch analysis, limit CSV files to **1,000 rows** for optimal Streamlit performance
- Confidence below 60% indicates borderline content; review the AI explanation carefully

---

<p align="center">
  <strong>Developed by <a href="https://github.com/issu321">issu321</a></strong>
</p>

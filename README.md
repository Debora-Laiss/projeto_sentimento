# Sentiment Analysis with Naive Bayes

This project implements a sentiment classification pipeline using a Naive Bayes model trained on labeled review datasets. The goal is to predict whether a given sentence has a **positive** or **negative** sentiment.

## 📁 Datasets

The following labeled datasets are used (sentence + sentiment):
- `amazon_cells_labelled.txt`
- `imdb_labelled.txt`
- `yelp_labelled.txt`

Each file contains lines formatted as:

Where:
- `sentence`: a short text (usually a product or service review)
- `label`: `1` for positive sentiment, `0` for negative sentiment

## 🧪 Pipeline Steps

1. **Load datasets** and concatenate them into a single DataFrame
2. **Preprocess text** using TF-IDF vectorization (removing English stop words)
3. **Split data** into training (80%) and testing (20%) sets
4. **Train model** using Multinomial Naive Bayes
5. **Evaluate performance** with accuracy score
6. **Generate predictions** for all samples
7. **Export results** to a CSV file for Power BI or further analysis

## 📊 Output

The script generates a file named:

## Resultado_sentimentos.csv

This CSV contains the original sentence, true sentiment, predicted sentiment, and a human-readable sentiment label (`positivo` or `negativo`).

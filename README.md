# Sentiment Analysis on Google Play Store Reviews

## Project Overview
This project focuses on performing sentiment analysis on user reviews collected from Google Play Store. The objective is to classify the reviews into positive and negative sentiments using various machine learning algorithms and deploy the final model as an interactive web application using Streamlit.

## Objectives
- Preprocess and clean user review text.
- Explore the dataset using basic visualizations and word frequency analysis.
- Train and compare multiple machine learning models.
- Build a user-friendly sentiment prediction tool using Streamlit.

## Dataset Description
The dataset contains mobile application reviews from the Google Play Store with two main columns:
- `review`: user-written text
- `sentiment`: label (positive or negative)

## Exploratory Data Analysis
- Sentiment distribution visualization
- Word cloud based on all review text
- Frequent terms and review length analysis

## Modeling and Deployment
Several machine learning models were trained and evaluated, including Naive Bayes, Random Forest, Logistic Regression, Decision Tree, and Support Vector Machine (SVM). The best-performing model was deployed using Streamlit.

The Streamlit app allows users to:
- Enter a review and receive sentiment prediction
- View sentiment distribution and review-based word cloud

## Tools and Technologies
- Python
- Pandas and NumPy for data handling
- Scikit-learn for machine learning
- NLTK and Sastrawi for text preprocessing
- Matplotlib and Seaborn for visualization
- WordCloud for word frequency analysis
- Streamlit for web deployment

## Key Learnings
- Applied end-to-end text preprocessing: case folding, stopword removal, tokenization, and stemming
- Compared performance of different machine learning models on sentiment classification
- Built and deployed a simple, interactive web application for text classification

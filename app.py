import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
import nltk
import re
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Load dataset
def load_data():
    return pd.read_csv('data_scrape.csv')

# Fungsi untuk menghasilkan WordCloud
def generate_wordcloud(data):
    tokenized_text = ' '.join(data['content'].dropna())
    wordcloud = WordCloud(width=300, height=150, background_color='white').generate(tokenized_text)
    plt.figure(figsize=(6, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('WordCloud of Most Frequent Words')
    st.image(wordcloud.to_array(), caption='WordCloud', use_column_width=True)

# Fungsi untuk menampilkan distribusi sentimen
def display_sentiment_distribution(data):
    sentiment_counts = data['score'].value_counts()

    fig_sentiment = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                           title='Sentiment Distribution')
    return fig_sentiment

# Fungsi untuk menampilkan distribusi waktu
def display_time_distribution(data):
    data['at'] = pd.to_datetime(data['at'])
    time_distribution = data['at'].dt.hour.value_counts().sort_index()

    fig_time_distribution = px.line(time_distribution, x=time_distribution.index, y=time_distribution.values,
                                    labels={'x': 'Hour', 'y': 'Count'}, title='Time Distribution')
    return fig_time_distribution

# Fungsi untuk menampilkan top 5 komentar
def display_top_comments(data):
    top_comments = data.nlargest(5, 'thumbsUpCount')[['userName', 'thumbsUpCount']]
    return top_comments

# Fungsi untuk menampilkan pie chart sentimen dengan thumbs up terbanyak
def display_sentiment_with_most_thumbsup_piechart(data):
    sentiment_thumbsup = data.groupby('score')['thumbsUpCount'].sum().reset_index()
    fig_sentiment_thumbsup = px.pie(sentiment_thumbsup, values='thumbsUpCount', names='score',
                                    title='Sentiment with Most ThumbsUpCount')
    fig_sentiment_thumbsup.update_layout(height=300, width=400)
    return fig_sentiment_thumbsup

# Fungsi untuk prediksi sentimen
def predict_sentiment(text, model, vectorizer):
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    return prediction

# Fungsi untuk menampilkan hasil klasifikasi
def display_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.write(df_report)

# Main function
def main():
    st.title('App Review Analysis')

    # Load data
    data = load_data()

    # Create choice
    choice = st.sidebar.selectbox("Select Choice", ["Sentiment Distribution", "Time Distribution", "WordCloud",
                                                    "Top 5 Comments", "Sentiment ThumbsUp", "Predict Sentiment"])

    if choice == "Sentiment Distribution":
        st.header('Sentiment Distribution')
        fig_sentiment = display_sentiment_distribution(data)
        st.plotly_chart(fig_sentiment)

    elif choice == "Time Distribution":
        st.header('Time Distribution')
        fig_time_distribution = display_time_distribution(data)
        st.plotly_chart(fig_time_distribution)

    elif choice == "WordCloud":
        st.header('WordCloud of Most Frequent Words')
        generate_wordcloud(data)

    elif choice == "Top 5 Comments":
        st.header('Top 5 Comments with Most ThumbsUpCount')
        top_comments = display_top_comments(data)
        st.dataframe(top_comments)

        st.header('Bar Chart for Top 5 Comments with Most ThumbsUpCount')
        fig_top_comments = px.bar(top_comments, x='userName', y='thumbsUpCount',
                                  labels={'thumbsUpCount': 'ThumbsUpCount', 'userName': 'User'},
                                  title='Top 5 Comments with Most ThumbsUpCount')
        fig_top_comments.update_layout(xaxis_title='User', yaxis_title='ThumbsUpCount', height=400)
        st.plotly_chart(fig_top_comments)

    elif choice == "Sentiment ThumbsUp":
        st.header('Sentiment with Most ThumbsUpCount')
        fig_sentiment_thumbsup = display_sentiment_with_most_thumbsup_piechart(data)
        st.plotly_chart(fig_sentiment_thumbsup)

    elif choice == "Predict Sentiment":
        st.header('Predict Sentiment')
        data = pd.read_csv('data_hasil_TextPreProcessing.csv')
        # Load additional data for sentiment prediction (e.g., cleaned text and labels)
        X = data['data_clean']  # Replace with actual column name of cleaned text
        y = data['label']  # Replace with actual column name of sentiment label

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Vectorize text using TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        # Choose a classification model (e.g., RandomForestClassifier)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_vectorized, y_train)

        # Take user input for prediction
        user_input = st.text_area("Enter your review here:")
        if st.button("Predict"):
            if user_input.strip() == "":
                st.warning("Please enter some text.")
            else:
                prediction = predict_sentiment(user_input, model, vectorizer)
                st.write(f"Predicted Sentiment: {prediction}")

                # Display classification report
                st.subheader("Classification Report")
                y_pred = model.predict(X_test_vectorized)
                display_classification_report(y_test, y_pred)

if __name__ == '__main__':
    main()

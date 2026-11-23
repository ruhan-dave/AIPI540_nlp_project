#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model training and prediction script.
This script handles the training of the sentiment analysis models and provides prediction functions.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_models(data_path):
    """
    Train sentiment analysis models on the preprocessed data.
    
    Args:
        data_path: Path to the preprocessed data file
        
    Returns:
        Trained models and vectorizers
    """
    # Load preprocessed data
    df = pd.read_csv(data_path)
    
    # Prepare data for training
    # Train on the *cleaned* text: the app scores cleaned tokens (lemmatised,
    # stop-worded) against this model's feature vocabulary. Training on the raw
    # `content` would put raw tokens ("isn't", "crashing") in the vocabulary and
    # almost every lookup in calculate_score_lr / calculate_score_nb would miss,
    # producing a blank, useless highlight.
    #
    # Reviews whose text is nothing but stopwords/short tokens clean to an empty
    # string; to_csv writes that as an empty field and read_csv reads it back as
    # NaN, which the vectorizers reject. Fill it and drop those rows — they carry
    # no signal for either vectorizer.
    df['cleaned_content'] = df['cleaned_content'].fillna('')
    df = df[df['cleaned_content'].str.strip() != '']

    X = df['cleaned_content']
    y = df['sentiment']
    
    # Create vectorizers
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,1), stop_words='english')
    count_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1,1), stop_words='english')
    
    # Transform text data
    X_tfidf = tfidf_vectorizer.fit_transform(X)
    X_count = count_vectorizer.fit_transform(X)
    
    # Split data
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, stratify=y, random_state=0
    )
    X_train_count, X_test_count, _, _ = train_test_split(
        X_count, y, test_size=0.2, stratify=y, random_state=0
    )
    
    # Train logistic regression model
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=0)
    lr_model.fit(X_train_tfidf, y_train)
    lr_predictions = lr_model.predict(X_test_tfidf)
    print("\nLogistic Regression Model Performance:")
    print(classification_report(y_test, lr_predictions))
    
    # Train Naive Bayes model
    print("\nTraining Naive Bayes model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_count, y_train)
    nb_predictions = nb_model.predict(X_test_count)
    print("\nNaive Bayes Model Performance:")
    print(classification_report(y_test, nb_predictions))
    
    # Save models and vectorizers
    save_models(lr_model, tfidf_vectorizer, nb_model, count_vectorizer)
    
    return lr_model, tfidf_vectorizer, nb_model, count_vectorizer

def save_models(lr_model, tfidf_vectorizer, nb_model, count_vectorizer):
    """
    Save the trained models and vectorizers.
    
    Args:
        lr_model: Trained logistic regression model
        tfidf_vectorizer: TF-IDF vectorizer
        nb_model: Trained Naive Bayes model
        count_vectorizer: Count vectorizer
    """
    models_dir = os.path.join('models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save logistic regression model
    with open(os.path.join(models_dir, 'lr_model.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
    
    # Save TF-IDF vectorizer
    with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    # Save Naive Bayes model
    with open(os.path.join(models_dir, 'nb_model.pkl'), 'wb') as f:
        pickle.dump(nb_model, f)
    
    # Save Count vectorizer
    with open(os.path.join(models_dir, 'count_vectorizer.pkl'), 'wb') as f:
        pickle.dump(count_vectorizer, f)
    
    print("Models and vectorizers saved successfully.")

def predict(text, model_type='lr'):
    """
    Predict sentiment for a given text.
    
    Args:
        text: Input text for prediction
        model_type: Model type to use ('lr' for Logistic Regression, 'nb' for Naive Bayes)
        
    Returns:
        Prediction and probability scores
    """
    if model_type == 'lr':
        # Load logistic regression model and vectorizer
        with open(os.path.join('models', 'lr_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        with open(os.path.join('models', 'tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        # Load Naive Bayes model and vectorizer
        with open(os.path.join('models', 'nb_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        with open(os.path.join('models', 'count_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
    
    # Transform input text
    X = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get probability scores
    proba = model.predict_proba(X)[0]
    scores = {
        'positive': proba[1] if model.classes_[1] == 'positive' else proba[0],
        'negative': proba[0] if model.classes_[0] == 'negative' else proba[1]
    }
    
    return prediction, scores

def main():
    """
    Main function to train models.
    """
    processed_data_path = os.path.join('data', 'processed', 'processed_reviews.csv')
    
    if os.path.exists(processed_data_path):
        print("Training models...")
        train_models(processed_data_path)
    else:
        print(f"Error: Processed data file not found at {processed_data_path}")
        print("Please run the preprocessing script first.")

if __name__ == "__main__":
    main() 
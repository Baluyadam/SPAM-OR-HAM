# Spam or Ham Classifier

This repository contains a Jupyter Notebook for classifying text messages as either Spam or Ham (not spam) using machine learning techniques.

## Dataset
The dataset used in this project consists of labeled text messages, with "spam" indicating unsolicited messages and "ham" indicating legitimate ones.

## Features & Methods
- **Text Preprocessing:** Tokenization, stopword removal, stemming/lemmatization.
- **Feature Extraction:** BOW vectorization.
- **Model Training:** Naive Bayes, Logistic Regression, or other classification algorithms.
- **Evaluation:** Accuracy.

## Requirements
Make sure you have the following installed:
- Python 3.x
Required libraries:
```bash
pip install streamlit pandas numpy scikit-learn
``` 
## Installation & Usage
1. Clone this repository or download the source code.
2. Navigate to the project directory and ensure dependencies are installed.
3. Run the Streamlit app using:
```bash
streamlit run app.py
``` 
4. Upload the spam.csv dataset and select a model to train.
5. Enter an email to classify it as Spam or Ham.
## Dataset
The application expects a dataset in CSV format with the following columns:
- Category: Label indicating spam (1) or ham (0)
- Message: Email text content
Ensure the dataset follows this format before uploading.
## File Structure
- ## Main Streamlit application
- ├── classi.py    
- ## Sample dataset
- ├── spam.csv              
- ## Jupyter Notebook with model analysis
- ├── Spam OR Ham.ipynb
- ## Project documentation
- ├── README.md             


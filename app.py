import streamlit as st 
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("spam.csv")
st.title(":blue[Classification of Email(Spam/Ham)]") 

x = df["Message"]
y = df["Category"]

bow = CountVectorizer (stop_words = "english")

final_X = pd.DataFrame (bow.fit_transform(x).toarray(), columns = bow.get_feature_names())
X_train, X_test, y_train, y_test = train_test_split(final_X, y , test_size= 0.25, random_state = 23)

models={"KNN": KNeighborsClassifier(),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVC": SVC(),
    "Logistic Regression": LogisticRegression()}
Select_Algorithm=st.selectbox("Select Algorithm",list(models.keys()))
obj = models[Select_Algorithm]

obj.fit(X_train,y_train)

y_pred = obj.predict(X_test)
res = st.button("Accuracy Test")

if res:
     st.write(accuracy_score(y_test, y_pred))
input = st.text_input("Enter email Text")

def fun(email):
    data=bow.transform([email]).toarray()
    st. write(obj.predict(data)[0])

if st.button("Predict Spam/Ham"):
    fun(input)
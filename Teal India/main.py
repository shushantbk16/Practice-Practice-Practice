import pandas as pd 
import re

train_data=pd.read_excel('task_dataset.xlsx' ,sheet_name="training_dataset")
val_data=pd.read_excel('task_dataset.xlsx' ,sheet_name="validation_dataset")
# print(val_data.head())


def clean_data(data):
    data=data.lower()
    data=re.sub(r'\b(no\.?|num)\b','number',data)
    data=re.sub(r'\b(flt|apt)\b','flat',data)
    data=re.sub(r'\b(h\.?no)\b','flat',data)
    data=re.sub(r'[^a-z0-9\s\s\/]',' ',data)
    data=re.sub(r'\s+',' ',data).strip()
    return data



# train_data['clean_text'] = train_data.iloc[:,0].apply(clean_data)
# print(train_data['clean_text'].head())

import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline




def train():
    training_data=train_data.iloc[:,0].apply(clean_data)
    validation_data=val_data.iloc[:,0].apply(clean_data)
    pipeline=Pipeline([
        ('tfidf',TfidfVectorizer(max_features=5000,ngram_range=(1,2))),
        ('clf',LogisticRegression(class_weight='balanced',max_iter=1000))
    ])
    pipeline.fit(training_data,train_data.iloc[:,1])
    y_pred=pipeline.predict(validation_data)
    print(classification_report(val_data.iloc[:,1],y_pred,))




if __name__=="__main__":
    import sklearn
    print(f"Scikit-Learn Version: {sklearn.__version__}")
    # train()
    
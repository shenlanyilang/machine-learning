import numpy as np
import pandas as pd
import os
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
import time

lematizer = WordNetLemmatizer()
stoplist = stopwords.words("english")
stopSet = set(stoplist)
all_data_dir = "./all_cleaned_data"


def my_tokenize(text: str) -> list:
    text = re.sub(r"[^a-z_]", r" ", text.lower().strip())
    text = re.sub(r" +", r" ", text.strip())
    textList = text.split(" ")
    textList = [lematizer.lemmatize(word, "n") for word in textList]
    textList = [word for word in textList if word not in stopSet]
    return textList


def lr_model(train_features, train_labels, test_features):
    lr_best = LogisticRegression(multi_class="multinomial", solver="sag", random_state=20,
                            C=10.0, penalty="l2")

    lr_best.fit(X=train_features, y=train_labels)
    train_preds = lr_best.predict(train_features)
    test_preds = lr_best.predict(test_features)
    return train_preds, test_preds, lr_best

def nb_model(train_features, train_labels, test_features):
    nb_best = MultinomialNB(alpha=0.03)
    nb_best.fit(X=train_features, y=train_labels)
    train_preds = nb_model.predict(train_features)
    test_preds = nb_model.predict(test_features)
    return train_preds, test_preds, nb_best

def xgb_model(train_features, train_labels, test_features):
    xgb_best = xgb.XGBClassifier(n_estimators=150, objective="multi:softmax",
                                 learning_rate=0.1, reg_alpha=0.01,reg_lambda=0.1,max_depth=4,
                                 booster="gblinear", nthread=2, subsample=0.8, colsample_bytree=0.8,
                                 colsample_bylevel=0.8, random_state=20)
    xgb_best.fit(X=train_features, y=train_labels)
    train_preds = xgb_best.predict(train_features)
    test_preds = xgb_best.predict(test_features)
    return train_preds, test_preds, xgb_best

def prob2label(preds):
    labels = [a.tolist().index(max(a)) for a in preds]
    return labels

def output_metrics(model_name, train_labels, train_preds, test_labels, test_preds):
    print(model_name+" predict result is:")
    train_accuracy = accuracy_score(train_labels, train_preds)
    test_accuracy = accuracy_score(test_labels, test_preds)
    train_macro_p = precision_score(train_labels, train_preds, average="macro")
    test_macro_p = precision_score(test_labels, test_preds, average="macro")
    train_macro_r = recall_score(train_labels, train_preds, average="macro")
    test_macro_r = recall_score(test_labels, test_preds, average="macro")
    train_f1 = f1_score(train_labels, train_preds, average="macro")
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    print("train accuracy is ", train_accuracy)
    print("test accuracy is ", test_accuracy)
    print("train precision is ", train_macro_p)
    print("test precision is ", test_macro_p)
    print("train recall is ", train_macro_r)
    print("test recall is ", test_macro_r)
    print("train f1 is ", train_f1)
    print("test f1 is ", test_f1)




def main():
    file2cat = pickle.load(open("./parameters/file2cat.pkl", "rb"))
    file_list = os.listdir(all_data_dir)
    filenames = [all_data_dir + "/" + name for name in file_list]

    labels = [file2cat[f] for f in file_list]
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    indexs = list(range(len(labels)))
    random.seed(20)
    random.shuffle(indexs)

    split_factor = 0.8
    train_indexs = indexs[:int(split_factor * len(labels))]
    test_indexs = indexs[int(split_factor * len(labels)):]

    print("start running tf-idf feature extraction...")
    tfidf = TfidfVectorizer(
        input="filename",
        lowercase=True,
        tokenizer=my_tokenize,
        analyzer="word",
        stop_words=stoplist,
        ngram_range=(1, 3),
        max_df=0.9,
        min_df=3,
        #max_features=100000,
        norm="l2",
        sublinear_tf=False
    )

    all_data_features = tfidf.fit_transform(filenames)
    all_data_features = all_data_features.tocsr()
    print("tf-idf has been done...")
    print("features number is ", all_data_features.shape[1])

    train_features = all_data_features[train_indexs, :]
    train_labels = labels[train_indexs]
    test_features = all_data_features[test_indexs, :]
    test_labels = labels[test_indexs]

    train_preds, test_preds, nb_best = nb_model(train_features, train_labels, test_features)
    output_metrics("nb model", train_labels, train_preds, test_labels, test_preds)
    train_preds, test_preds, lr_best = lr_model(train_features, train_labels, test_features)
    output_metrics("lr model", train_labels, train_preds, test_labels, test_preds)
    train_preds, test_preds, xgb_best = xgb_model(train_features, train_labels, test_features)
    output_metrics("xgb model", train_labels, train_preds, test_labels, test_preds)


if __name__ == "__main__":
    main()
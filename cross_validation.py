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

xgb_cv_params = {
    "max_depth": [4, 5, 6],
    "reg_lambda":[0.1, 0.2],
    "learning_rate": [0.05, 0.1, 0.2],
    "reg_alpha": [0.05, 0.02, 0.01]
}

nb_cv_params = {
    "alpha": [0.03, 0.04, 0.05, 0.1]
}

lr_cv_params = {"penalty": ["l2"], "C": [7.0, 8.0, 9.0, 10.0]}

xgb_parames = {
    "max_depth":4,
    "reg_lambda": 0.1,
    "reg_alpha": 0.01,
    "learning_rate": 0.1,
    "n_estimators": 150,
    "objective":"multi:softmax"

}

def my_tokenize(text: str) -> list:
    text = re.sub(r"[^a-z_]", r" ", text.lower().strip())
    text = re.sub(r" +", r" ", text.strip())
    textList = text.split(" ")
    textList = [lematizer.lemmatize(word, "n") for word in textList]
    textList = [word for word in textList if word not in stopSet]
    return textList

def cv_model(model, params, X, y):
    cross_validator = KFold(n_splits=10)
    scoring_fnc = make_scorer(accuracy_score)
    grid = GridSearchCV(model, params, scoring=scoring_fnc, cv=cross_validator)
    grid.fit(X, y)
    return grid.best_params_, grid.best_estimator_

def prob2label(preds):
    labels = [a.tolist().index(max(a)) for a in preds]
    return labels

def getBestParams(train_features, train_labels, test_features, test_labels):
    start_time = time.time()
    nb_cv = MultinomialNB()
    nb_best_params, nb_best_model = cv_model(nb_cv, nb_cv_params, X=train_features, y=train_labels)
    print("nb best params are:")
    print(nb_best_params)
    train_preds = nb_best_model.predict(train_features)
    test_preds = nb_best_model.predict(test_features)
    output_metrics("nb best model", train_labels, train_preds, test_labels, test_preds)
    end_time = time.time()
    print("cost time ", end_time-start_time, "s")

    start_time = time.time()
    lr_cv = LogisticRegression(multi_class="multinomial", solver="sag", random_state=20)
    lr_best_params, lr_best = cv_model(lr_cv, lr_cv_params, train_features, train_labels)
    print("lr best params are: ")
    print(lr_best_params)
    train_preds = lr_best.predict(train_features)
    test_preds = lr_best.predict(test_features)
    output_metrics("lr best model", train_labels, train_preds, test_labels, test_preds)
    end_time = time.time()
    print("cost time ", end_time-start_time, "s")

    start_time = time.time()
    xgb_cv = xgb.XGBClassifier(n_estimators=150, objective="multi:softmax",
                               booster="gblinear", nthread=2, subsample=0.8, colsample_bytree=0.8,
                               colsample_bylevel=0.8, random_state=20)
    xgb_best_params, xgb_best_model = cv_model(xgb_cv, xgb_cv_params, X=train_features, y=train_labels)
    print("xgb best params are: ")
    print(xgb_best_params)
    train_preds = xgb_best_model.predict(train_features)
    test_preds = xgb_best_model.predict(test_features)
    output_metrics("xgb best model", train_labels, train_preds, test_labels, test_preds)
    end_time = time.time()
    print("cost time ", end_time-start_time, "s")

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

    getBestParams(train_features, train_labels, test_features, test_labels)


if __name__ == "__main__":
    main()
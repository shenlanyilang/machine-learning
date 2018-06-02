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
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import time

lematizer = WordNetLemmatizer()
stoplist = stopwords.words("english")
stopSet = set(stoplist)
n_components = 1000

lgb_params = {
    "application": "multiclass",
    "boosting": "gbdt",
    "num_boost_round": 400,
    "learning_rate": 0.2,
    "num_leaves": 20,
    "num_threads": 2,
    "max_depth": 5,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.9,
    "early_stopping_round": 300,
    "num_class": 20,
    "metric": "multi_logloss"

}

xgb_params ={
    "booster":"gblinear",
    "nthread":2,
    "max_depth":5,
    "subsample":0.8,
    "colsample":0.8,
    "lambda": 2.0,
    "alpha": 1,
    "objective": "multi:softmax",
    "num_class": 20,
    "eval_metric": "mlogloss",
    "seed":20
}

xgb_cv_params = {
    "max_depth": [4, 5, 6],
    "reg_lambda":[0.1, 0.2],
    "learning_rate": [0.05, 0.1, 0.2],
    "reg_alpha": [0.05, 0.02, 0.01]
}

nb_cv_params = {
    "alpha": [0.001,0.002,0.003,0.004,0.005,0.01]
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


def svc_cv_model(train_features, train_labels, test_features, test_labels):
    # best params: kernel:linear, C=2.0
    svc_model = SVC(random_state=20)
    svc_params = {"kernel": ["linear", "rbf"], "C": [0.2, 0.5, 1.0, 2.0, 4.0]}
    best_params, svc_best = cv_model(svc_model, svc_params, train_features, train_labels)
    print("best params are:")
    print(best_params)
    train_preds = svc_best.predict(train_features)
    test_preds = svc_best.predict(test_features)
    train_accuracy = accuracy_score(train_labels, train_preds)
    test_accuracy = accuracy_score(test_labels, test_preds)
    print("svm results...")
    print("train accuracy is ", train_accuracy)
    print("test accuracy is ", test_accuracy)
    return best_params, svc_best


def lr_cv_model(train_features, train_labels, test_features, test_labels):
    # best params l2,C=5
    lr_model = LogisticRegression(multi_class="multinomial", solver="sag", random_state=20)
    lr_params = {"penalty": ["l2"], "C": [2.0, 3.0, 4.0, 5.0, 6.0]}
    best_params, lr_best = cv_model(lr_model, lr_params, train_features, train_labels)
    print("best params are:")
    print(best_params)
    train_preds = lr_best.predict(train_features)
    test_preds = lr_best.predict(test_features)
    train_accuracy = accuracy_score(train_labels, train_preds)
    test_accuracy = accuracy_score(test_labels, test_preds)
    print("logistic regression results...")
    print("regression train accuracy is ", train_accuracy)
    print("test accuracy is ", test_accuracy)
    return best_params, lr_best


# def svd_comp(features):
#    [u, s, vt] = svds(features, k=n_components)

def lda_model(features):
    lda = LatentDirichletAllocation(n_components=n_components, random_state=20,
                                    learning_method="batch")
    new_features = lda.fit_transform(features)
    return new_features


def lr_model(train_features, train_labels, test_features, test_labels):
    lr = LogisticRegression(multi_class="multinomial", solver="sag", random_state=20,
                            C=5.0, penalty="l2")

    lr.fit(X=train_features, y=train_labels)
    train_preds = lr.predict(train_features)
    test_preds = lr.predict(test_features)
    train_accuracy = accuracy_score(train_labels, train_preds)
    test_accuracy = accuracy_score(test_labels, test_preds)
    train_macro_p = precision_score(train_labels, train_preds, average="macro")
    test_macro_p = precision_score(test_labels, test_preds, average="macro")
    train_macro_r = recall_score(train_labels, train_preds, average="macro")
    test_macro_r = recall_score(test_labels, test_preds, average="macro")
    train_f1 = f1_score(train_labels, train_preds, average="macro")
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    print("logistic regression results...")
    print("train accuracy is ", train_accuracy)
    print("test accuracy is ", test_accuracy)
    print("train precision is ", train_macro_p)
    print("test precision is ", test_macro_p)
    print("train recall is ", train_macro_r)
    print("test recall is ", test_macro_r)
    print("train f1 is ", train_f1)
    print("test f1 is ", test_f1)


def svc_model(train_features, train_labels, test_features, test_labels):
    svc_model = SVC(kernel="linear", C=2.0, random_state=20)
    svc_model.fit(X=train_features, y=train_labels)
    train_preds = svc_model.predict(train_features)
    test_preds = svc_model.predict(test_features)
    train_accuracy = accuracy_score(train_labels, train_preds)
    test_accuracy = accuracy_score(test_labels, test_preds)
    train_macro_p = precision_score(train_labels, train_preds, average="macro")
    test_macro_p = precision_score(test_labels, test_preds, average="macro")
    train_macro_r = recall_score(train_labels, train_preds, average="macro")
    test_macro_r = recall_score(test_labels, test_preds, average="macro")
    train_f1 = f1_score(train_labels, train_preds, average="macro")
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    print("svm results...")
    print("train accuracy is ", train_accuracy)
    print("test accuracy is ", test_accuracy)
    print("train precision is ", train_macro_p)
    print("test precision is ", test_macro_p)
    print("train recall is ", train_macro_r)
    print("test recall is ", test_macro_r)
    print("train f1 is ", train_f1)
    print("test f1 is ", test_f1)


def nb_model(train_features, train_labels, test_features, test_labels):
    nb_model = MultinomialNB()
    nb_model.fit(X=train_features, y=train_labels)
    train_preds = nb_model.predict(train_features)
    y_preds = nb_model.predict(test_features)

    train_accuracy = accuracy_score(train_labels, train_preds)
    accuracy = accuracy_score(test_labels, y_preds)
    print("train accuracy is ", train_accuracy)
    print("test accuracy is ", accuracy)

def prob2label(preds):
    labels = [a.tolist().index(max(a)) for a in preds]
    return labels

def lgb_model(train_features, train_labels, test_features, test_labels):
    print("lgb_model begins ...")
    start_time = time.time()
    train_nums = len(train_labels)
    split_fac = 0.9
    train_features, valid_features = train_features[:int(train_nums * split_fac), :], train_features[
                                                                                      int(train_nums * split_fac):, :]
    train_labels, valid_labels = train_labels[:int(train_nums * split_fac)], train_labels[int(train_nums * split_fac):]
    train_data = lgb.Dataset(data=train_features, label=train_labels)
    valid_data = lgb.Dataset(data=valid_features, label=valid_labels)
    lgb_result = lgb.train(params=lgb_params, train_set=train_data,
                           valid_sets=valid_data)

    end_time = time.time()
    print("lgb_model trains end, cost time " + str(end_time - start_time) + "s")
    train_preds = lgb_result.predict(data=train_features)
    train_preds = prob2label(train_preds)
    valid_preds = lgb_result.predict(data=valid_features)
    valid_preds = prob2label(valid_preds)

    test_preds = lgb_result.predict(data=test_features)
    test_preds = prob2label(test_preds)
    train_accuracy = accuracy_score(train_labels, train_preds)
    test_accuracy = accuracy_score(test_labels, test_preds)
    train_macro_p = precision_score(train_labels, train_preds, average="macro")
    test_macro_p = precision_score(test_labels, test_preds, average="macro")
    train_macro_r = recall_score(train_labels, train_preds, average="macro")
    test_macro_r = recall_score(test_labels, test_preds, average="macro")
    train_f1 = f1_score(train_labels, train_preds, average="macro")
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    print("lgbm results...")
    print("train accuracy is ", train_accuracy)
    print("test accuracy is ", test_accuracy)
    print("train precision is ", train_macro_p)
    print("test precision is ", test_macro_p)
    print("train recall is ", train_macro_r)
    print("test recall is ", test_macro_r)
    print("train f1 is ", train_f1)
    print("test f1 is ", test_f1)

def rf_model(train_features, train_labels, test_features):
    rf = RandomForestClassifier(
        n_estimators=500,
        #max_depth=6,
        min_samples_leaf=64,
        n_jobs=2,
        random_state=20
    )
    rf.fit(X=train_features, y=train_labels)
    train_preds = rf.predict(train_features)
    test_preds = rf.predict(test_features)
    return train_preds, test_preds

def xgb_model(train_features, train_labels, test_features):
    train_data = xgb.DMatrix(data=train_features, label=train_labels)
    test_data = xgb.DMatrix(data=test_features)

    xgb_result = xgb.train(params=xgb_params, dtrain=train_data,
                           num_boost_round=50, early_stopping_rounds=300,
                           learning_rates=[]
                           )
    train_preds = xgb_result.predict(train_data)
    train_preds = prob2label(train_preds)
    test_preds = xgb_result.predict(test_data)
    test_preds = prob2label(test_preds)
    return train_preds, test_preds


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
    train_dir = "./clean/20news-bydate-train"
    test_dir = "./clean/20news-bydate-test"
    train_cat = os.listdir(train_dir)
    test_cat = os.listdir(test_dir)
    all_data_dir = "./all_cleaned_data"
    if not os.path.exists(all_data_dir):
        os.mkdir(all_data_dir)
        file2cat = {}
        for cat in train_cat:
            files = os.listdir(os.path.join(train_dir, cat))
            for f in files:
                file2cat[cat+f] = cat
                shutil.copy(os.path.join(train_dir, cat, f), "./all_cleaned_data"+"/"+cat+f)

        for cat in test_cat:
            files = os.listdir(os.path.join(test_dir, cat))
            for f in files:
                file2cat[cat+f] = cat
            shutil.copy(os.path.join(test_dir, cat, f), "./all_cleaned_data"+"/"+cat+f)

        if not os.path.exists("./parameters"):
            os.mkdir("./parameters")
            fp = open("./parameters/file2cat.pkl", "wb")
            pickle.dump(file2cat, fp)
            fp.close()

    # all_data_dir = "./original_files"
    file2cat = pickle.load(open("./parameters/file2cat.pkl", "rb"))
    file_list = os.listdir(all_data_dir)
    filenames = [all_data_dir + "/" + name for name in file_list]

    labels = [file2cat[f] for f in file_list]
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    indexs = list(range(len(labels)))
    random.shuffle(indexs)

    split_factor = 0.8
    train_indexs = indexs[:int(split_factor * len(labels))]
    test_indexs = indexs[int(split_factor * len(labels)):]

    count_vectorizer = CountVectorizer(
        input="filename",
        analyzer="word",
        tokenizer=my_tokenize,
        ngram_range=(1, 1),
        lowercase=True,
        max_df=0.95,
        min_df=3,
        # max_features=100000
    )
    # print("count vectorizer begins...")
    # doc_cnt = count_vectorizer.fit_transform(filenames)
    # print("start running lda model...")
    # topic_features = csr_matrix(lda_model(doc_cnt))

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

    print("tf-idf has been done...")
    # tf_idf_features = csr_matrix(tfidf.fit_transform(filenames))
    # all_data_features = hstack([tf_idf_features, topic_features])
    all_data_features = tfidf.fit_transform(filenames)
    all_data_features = all_data_features.tocsr()
    print("features number is ", all_data_features.shape[1])
    # all_data_features = tfidf.fit_transform(filenames)

    train_features = all_data_features[train_indexs, :]
    train_labels = labels[train_indexs]
    test_features = all_data_features[test_indexs, :]
    test_labels = labels[test_indexs]
    # lr_model(train_features, train_labels, test_features, test_labels)
    #lgb_model(train_features, train_labels, test_features, test_labels)
    #train_preds, test_preds = rf_model(train_features, train_labels, test_features)
    #output_metrics("rf model", train_labels, train_preds, test_labels, test_preds)
    #train_preds, test_preds = xgb_model(train_features, train_labels, test_features)
    #output_metrics("xgboost model", train_labels, train_preds, test_labels, test_preds)

    best_cv_params = {'reg_alpha': 0.01, 'max_depth': 4, 'learning_rate': 0.1, 'reg_lambda': 0.1}
    #xgb_cv = xgb.XGBClassifier(n_estimators=150, objective="multi:softmax",
    #                           booster="gblinear", nthread=2, subsample=0.8, colsample_bytree=0.8,
    #                           colsample_bylevel=0.8, random_state=20)
    #xgb_best_params, xgb_best_model = cv_model(xgb_cv, xgb_cv_params, X=train_features, y=train_labels)
    #print("xgb best params are: ")
    #print(xgb_best_params)
    #train_preds = xgb_best_model.predict(train_features)
    #test_preds = xgb_best_model.predict(test_features)
    #output_metrics("xgb best model", train_labels, train_preds, test_labels, test_preds)
    best_cv_params = {"alpha":0.01}
    nb_cv = MultinomialNB()
    nb_best_params, nb_best_model = cv_model(nb_cv, nb_cv_params, X=train_features, y=train_labels)
    print(nb_best_params)
    train_preds = nb_best_model.predict(train_features)
    test_preds = nb_best_model.predict(test_features)
    output_metrics("nb best model", train_labels, train_preds, test_labels, test_preds)


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import defaultdict
import time
from collections import Counter
import shutil

stoplist = stopwords.words("english")
stopSet = set(stoplist)
n_components = 1000
lematizer = WordNetLemmatizer()

train_dir = "./clean/20news-bydate-train"
test_dir = "./clean/20news-bydate-test"
train_original_dir = "./20news-bydate-train"
test_original_dir = "./20news-bydate-test"
train_cat = os.listdir(train_dir)
test_cat = os.listdir(test_dir)
all_data_dir = "./all_cleaned_data"
# all_data_dir = "./original_files"
file2cat = pickle.load(open("./parameters/file2cat.pkl", "rb"))
file_list = os.listdir(all_data_dir)
filenames = [all_data_dir + "/" + name for name in file_list]

class MyDoc(object):

    def __init__(self):
        self.original_train = train_original_dir
        self.original_test = test_original_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.all_data_dir = all_data_dir


    def my_tokenize(self,text: str) -> list:
        text = re.sub(r"[^a-z_]", r" ", text.lower().strip())
        text = re.sub(r" +", r" ", text.strip())
        textList = text.split(" ")
        textList = [lematizer.lemmatize(word, "n") for word in textList]
        textList = [word for word in textList if word not in stopSet]
        return textList

    def getTextLen(self, filepath):
        f1 = open(filepath,"rb")
        s = str(f1.read())
        f1.close()
        s = s.replace("\\n"," ")
        s = re.sub(r" +"," ",s)
        s = s.split(" ")
        return s

    def getDocLen(self):
        flen_list = []
        for d in [self.original_train, self.original_test]:
            for root, dirs, files in os.walk(d):
                for name in files:
                    word_list =self.getTextLen(os.path.join(root, name))
                    flen_list.append(len(word_list))
        return flen_list

    def getWordCnt(self):
        word_cnt = Counter()
        for file in os.listdir(self.all_data_dir):
            content = open(os.path.join(all_data_dir, file), "r").read()
            word_list = self.my_tokenize(content)
            for word in word_list:
                word_cnt[word] += 1
        return word_cnt

    def getCatCnt(self):
        cat_cnt = defaultdict(int)
        for file, cat in file2cat.items():
            cat_cnt[cat] += 1
        return cat_cnt

    def catDocLen(self):
        catLen = defaultdict(list)
        catMeanLen = {}
        for file in os.listdir(self.all_data_dir):
            catLen[file2cat[file]].append(len(self.getTextLen(os.path.join(self.all_data_dir, file))))
        for cat, docList in catLen.items():
            catMeanLen[cat] = np.mean(np.array(docList))
        return catMeanLen

def main():
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

    data_etl = MyDoc()
    doc_cnt_len = data_etl.getDocLen()
    word_cnt = data_etl.getWordCnt()
    cat_cnt = data_etl.getCatCnt()
    cat_doc_len = data_etl.catDocLen()
    print("catogory word count is ", cat_cnt)
    print("most common 20 words are ", word_cnt.most_common(20))
    print("all doc mean length is ", np.mean(np.array(doc_cnt_len)))
    print("catogory mean word count is ", cat_doc_len)


if __name__ == "__main__":
    main()

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

data_etl = MyDoc()
doc_cnt_len = data_etl.getDocLen()
word_cnt = data_etl.getWordCnt()
cat_cnt = data_etl.getCatCnt()
cat_doc_len = data_etl.catDocLen()
print("catogory word count is ", cat_cnt)
print("most common 20 words are ", word_cnt.most_common(20))
print("all doc mean length is ", np.mean(np.array(doc_cnt_len)))
print("catogory mean word count is ", cat_doc_len)

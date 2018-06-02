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

stoplist = stopwords.words("english")
stopSet = set(stoplist)
n_components = 1000
lematizer = WordNetLemmatizer()

train_dir = "./clean/20news-bydate-train"
test_dir = "./clean/20news-bydate-test"
train_cat = os.listdir(train_dir)
test_cat = os.listdir(test_dir)
all_data_dir = "./all_cleaned_data"
# all_data_dir = "./original_files"
file2cat = pickle.load(open("./parameters/file2cat.pkl", "rb"))
file_list = os.listdir(all_data_dir)
filenames = [all_data_dir + "/" + name for name in file_list]

def my_tokenize(text: str) -> list:
    text = re.sub(r"[^a-z_]", r" ", text.lower().strip())
    text = re.sub(r" +", r" ", text.strip())
    textList = text.split(" ")
    textList = [lematizer.lemmatize(word, "n") for word in textList]
    textList = [word for word in textList if word not in stopSet]
    return textList

cat_doc_cnt = defaultdict(int)
for doc, cat in file2cat.items():
    cat_doc_cnt[cat] += 1

print(cat_doc_cnt)
print(sum(cat_doc_cnt.values()))
cat_words_cnt = defaultdict(list)
vocb = set()
#for file in filenames:
#    with open(file, "r") as f1:
#        content = f1.read()
#        conlist = my_tokenize(content)
#        cat_words_cnt[file2cat[file.split("/")[-1]]].append(len(set(conlist)))
#        vocb.update(conlist)
#
#print("word number is ", len(vocb))
#all_word_cnt = []
#vocb_size = len(vocb)
#for cat, word_cnt in cat_words_cnt.items():
#    print("catagory "+cat+"word cnt distibution like...")
#    plt.hist(word_cnt, bins=40)
#    time.sleep(3)
#    all_word_cnt += word_cnt
#
#print("all doc word distribution like...")
#plt.hist(all_word_cnt, bins=40)


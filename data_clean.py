import numpy as np
import pandas as pd
import re
import os
import shutil

def cleanTokens(text):
    newText = re.sub(r"[^0-9a-zA-Z,:\.@'\"()-_!?\n]+|\*+", " ", text)
    newText = re.sub(r"(\\t)+|(\\f)+|(\\v)+|(\\r)+", " ", newText)
    newText = re.sub(r"[\r\t\f\v]+", " ", newText)
    newText = re.sub(r"[<>^]+", " ", newText)
    newText = re.sub(r"\\(')", r"\1", newText)
    newText = re.sub(r"[- ]+\n", "\n", newText)
    newText = re.sub(r"\n[ -]+", "", newText)
    newText = re.sub(r"\n+", "\n", newText)
    newText = re.sub(r" +", " ", newText)
    return newText

def main():
    train_old_path = "./20news-bydate-train"
    test_old_path = "./20news-bydate-test"
    path1 = "./clean/20news-bydate-train"
    path2 = "./clean/20news-bydate-test"
    if os.path.exists("./clean"):
        shutil.rmtree("./clean")
    os.makedirs(path1)
    os.makedirs(path2)
    train_cat = os.listdir(train_old_path)
    test_cat = os.listdir(test_old_path)
    for cat in train_cat:
        path_old = os.path.join(train_old_path,cat)
        path_new = os.path.join(path1,cat)
        os.makedirs(path_new)
        files = os.listdir(path_old)
        for f in files:
            with open(os.path.join(path_old,f), "rb") as f1:
                content = f1.read()
                content = str(content)[2:-1]
                content = re.sub(r"(\\n)+", "\n", content)
                newContent = cleanTokens(content)
            with open(os.path.join(path_new, f), "w") as f2:
                f2.write(newContent)
    for cat in test_cat:
        path_old = os.path.join(test_old_path,cat)
        path_new = os.path.join(path2,cat)
        os.makedirs(path_new)
        files = os.listdir(path_old)
        for f in files:
            with open(os.path.join(path_old,f), "rb") as f1:
                content = f1.read()
                content = str(content)[2:-1]
                content = re.sub(r"\\n+", "\n", content)
                newContent = cleanTokens(content)
            with open(os.path.join(path_new, f), "w") as f2:
                f2.write(newContent)
if __name__ == "__main__":
    main()

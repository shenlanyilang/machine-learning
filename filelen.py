import os
import re
import numpy as np

def getTextLen(filepath):
    f1 = open(filepath,"rb")
    s = str(f1.read())
    f1.close()
    s = s.replace("\\n"," ")
    s = re.sub(r" +"," ",s)
    s = s.split(" ")
    return s

train_dir = "./20news-bydate-train"
test_dir = "./20news-bydate-test"
flen_list = []
for d in [train_dir, test_dir]:
    for root, dirs, files in os.walk(d):
        for name in files:
            word_list = getTextLen(os.path.join(root, name))
            flen_list.append(len(word_list))
print(np.mean(np.array(flen_list)))

#os.chdir("20news-bydate-train")
#print(os.getcwd())
#cdirs = os.listdir()
#flen_dict = {}
#for cdir in cdirs:
#    flen_list = []
#    files = os.listdir(cdir)
#    for f in files:
#        filepath = os.path.join(cdir,f)
#        f_lengh = getTextLen(filepath)
#        flen_list.append(len(f_lengh))
#    flen_dict[cdir] = np.array(flen_list).mean()
#
#print(flen_dict)




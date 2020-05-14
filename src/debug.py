import glob
import os
print(os.getcwd())
a=glob.glob("./saved_models/seq2seq/*/")
versions=[i.split("/")[-2] for i in a]

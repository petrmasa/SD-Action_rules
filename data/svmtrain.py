import pandas as pd
import os
from sklearn.impute import SimpleImputer

dir = os.getcwd()

os.makedirs("data_pkl",exist_ok=True)

dir = os.path.join(dir,"data_pkl")

from pandas.api.types import CategoricalDtype
def getlabels(s):
    lst = []
    for i in range(len(s)-1):
        j=i+1
        if (j==1):
            item = '<' + str(s[i]) + ','+ str(s[i+1]) + '>'
#            item = ' ' + str(j) + ' : <' + str(s[i]) + ','+ str(s[i+1]) + '>'
        else:
            item = '(' + str(s[i]) + ','+ str(s[i+1]) + '>'
#            item = '' + str(j) + ' : (' + str(s[i]) + ',' + str(s[i + 1]) + '>'
        lst.append(item)

    print(s)
    print(lst)
    return lst



original=pd.read_csv("data_src/SVMtrain.csv")#, sep='\t')

print(original.columns)
#exit(0)

#original['Education'] = original['Education'].astype('category').cat.reorder_categories(edu_cat,ordered=True)

to_qcut=[ 'Age', 'Fare']


for varname in to_qcut:
    original[varname] = pd.qcut(original[varname], q=5,duplicates='drop')


df=original

df = df.drop('PassengerId', axis=1)
fname = os.path.join(dir,'svmtrain.zip')
df.to_pickle(fname,compression='zip')
print(f"Your file is in {fname}")

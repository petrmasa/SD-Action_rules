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
        else:
            item = '(' + str(s[i]) + ','+ str(s[i+1]) + '>'
        lst.append(item)

    print(s)
    print(lst)
    return lst



features = ["Age", "Workclass", "fnlwgt", "Education", "Education_Num", "Martial_Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital_Gain", "Capital_Loss",
        "Hours_per_week", "Country", "Target"]



edu_cat = ["Preschool","1st-4th", "5th-6th", "7th-8th","9th","10th","11th","12th","HS-grad","Some-college","Assoc-voc","Assoc-acdm","Bachelors","Masters","Prof-school","Doctorate"]
edu_cat_type =CategoricalDtype(categories=edu_cat, ordered=True)


train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'




original_train = pd.read_csv(train_url, names=features, sep=r'\s*,\s*',
                             engine='python', na_values="?")
original_test = pd.read_csv(test_url, names=features, sep=r'\s*,\s*',
                            engine='python', na_values="?", skiprows=1)

original_test.Target = original_test.Target.str.replace('.','')


original = pd.concat([original_train, original_test])

original['Education'] = original['Education'].astype('category').cat.reorder_categories(edu_cat,ordered=True)

age_bins=[10,20,30,40,50,60,70,90]
original['Age_b'] = pd.cut(original['Age'], include_lowest=True, bins = age_bins, labels = getlabels(age_bins), ordered=True)
hpw_bins=[0,10,20,30,40,50,60,70,100]
original['Hours_per_week_b'] = pd.cut(original['Hours_per_week'], include_lowest=True, bins = hpw_bins, labels = getlabels(hpw_bins), ordered=True)
cl_bins=[-1,0,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,4400]
original['Capital_Loss_b'] = pd.cut(original['Capital_Loss'], include_lowest=True, bins = cl_bins,labels = getlabels(cl_bins), ordered=True)
cg_bins = [-1,0,2000,3000,4000,5000,7000,10000,20000,99000,100000]
original['Capital_Gain_b'] = pd.cut(original['Capital_Gain'], include_lowest=True, bins = cg_bins , labels = getlabels(cg_bins), ordered=True)

original['Income']=original['Target']

df = original[['Income','Capital_Gain_b','Capital_Loss_b','Hours_per_week_b','Occupation','Martial_Status','Relationship','Age_b','Education','Sex','Country','Race','Workclass','Target']]


df = df.reset_index(drop=True)
fname = os.path.join(dir,'adults.zip')
df.to_pickle(fname,compression='zip')
print(f"Your file is in {fname}")

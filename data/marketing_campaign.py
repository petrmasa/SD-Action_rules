import pandas as pd
import os
from sklearn.impute import SimpleImputer

dir = os.getcwd()

os.makedirs("data_pkl",exist_ok=True)

dir = os.path.join(dir,"data_pkl")

from pandas.api.types import CategoricalDtype
original=pd.read_csv("data_src/marketing_campaign.csv", sep='\t')

print(original.columns)
#exit(0)

#original['Education'] = original['Education'].astype('category').cat.reorder_categories(edu_cat,ordered=True)

to_qcut=[ 'Year_Birth',  'Income', 'Kidhome',
       'Teenhome',  'Recency', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue']


for varname in to_qcut:
    original[varname] = pd.qcut(original[varname], q=5,duplicates='drop')


df=original
fname = os.path.join(dir,'marketing_campaign.zip')
df.to_pickle(fname,compression='zip')
print(f"Your file is in {fname}")

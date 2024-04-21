import pandas as pd
import os


dir = os.getcwd()

os.makedirs("data_pkl",exist_ok=True)

dir = os.path.join(dir,"data_pkl")

#df = pd.read_csv ('w:\\development\\cleverminer\\_data\\accidents.txt ', encoding='cp1250', sep='\t')

#df=df[['Driver_Age_Band','Driver_IMD','Sex','Journey','Hit_Objects_in','Hit_Objects_off','Casualties','Severity']]

#imputer = SimpleImputer(strategy="most_frequent")
#df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)


df = pd.read_csv("data_src/diabetes.csv")

print(df.columns)

for col in df.columns:
    dfc=df[col]
    cnt=len(dfc.unique())
    print(f"...col {col} has {cnt} unique values")
    if cnt>=20:
        df[col]=pd.qcut(df[col],10,duplicates="drop")
        print("...->updated")
        print(f"...->col {col} has {len(df[col].unique())} unique values : {df[col].unique()}")

fname = os.path.join(dir,'diabetes.zip')
df.to_pickle(fname,compression='zip')
print(f"Your file is in {fname}")

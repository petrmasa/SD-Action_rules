import math

import pandas as pd
import sys
import sklearn.impute

from sklearn.impute import SimpleImputer

from cleverminer import cleverminer

import scipy

#exmaple that shows the entire workflow from the basic flow on multiple datasets

import os
from pandas.api.types import CategoricalDtype

def action_rules(df=None,quantifiers=None,ante=None,succ=None,ante_var=None,cond=None):
    if (df is None or quantifiers is None or ante is None or succ is None or ante_var is None):
        print('Some attribute is missing. Please use all named attributes')
        exit(1)


    clm = cleverminer(df=df,proc='SD4ftMiner',quantifiers= quantifiers,
                      ante=ante,succ=succ,frst=ante_var, scnd=ante_var,cond=cond )
    return clm


dir = os.getcwd()


dir = os.path.join(dir,"data\\data_pkl")

ds = pd.read_csv(os.path.join(dir,'datasets.csv'),sep=';')

ds['rules_total']=None
ds['rules_act']=None

print(ds)

for ind in ds.index:
    ignore_var_list=[]
    print(type(ds['ignore_vars'][ind]))
    #if(type(ds['ignore_vars'][ind]))<>2:
    try:
        ignore_var_list = ds['ignore_vars'][ind].split("|")
    except:
        ignore_var_list=[]
    print(f"...will go for dataset {ds['filename'][ind]}, target {ds['target'][ind]}, target_class {ds['target_class'][ind]}, split_class {ds['split_var'][ind]}, ignore_vars {ds['ignore_vars'][ind]}, i.e. {ignore_var_list}")
    fname=ds['filename'][ind]
    df = pd.read_pickle(os.path.join(dir,fname))
    print(f"......will go for column {ds['target'][ind]} out of {df.columns}")
    tgt=ds['target'][ind]
    df2 = df[tgt]
    print(df2.unique())
    cls = ds['target_class'][ind]
    if cls.isnumeric():
        cls=int(cls)

    ante = []
    for itm in df.columns:
        if not(itm==ds['target'][ind] or itm==ds['split_var'][ind] or itm in ignore_var_list):
            ante_ln={}
            ante_ln['name']=itm
            ante_ln['type']='subset'
            ante_ln['minlen']=1
            ante_ln['maxlen'] = 1
            ante.append(ante_ln)


    print(f"...ante is {ante}")
    relbase = 0.01

    clm = action_rules(df=df,
                  quantifiers={'RelBase1': 0.01, 'RelBase2': 0.01, 'Ratiopim': 1.05},
                  ante={
                      'attributes': ante , 'minlen': 1, 'maxlen': 1, 'type': 'con'},
                  succ={
                      'attributes': [
                          {'name': ds['target'][ind], 'type': 'one', 'value': cls}
                      ], 'minlen': 1, 'maxlen': 1, 'type': 'con'},
                  ante_var={
                      'attributes': [
                          {'name': ds['split_var'][ind], 'type': 'subset', 'minlen': 1, 'maxlen': 1}

                      ], 'minlen': 1, 'maxlen': 1, 'type': 'con'}
                  )


    clm.print_rulelist()

    rules = clm.result.get('rules')

    ds['rules_total'][ind] = clm.get_rulecount()


    newrules=[]

    cnt=0
    cnt_filtered_out=0

    rows= len(df)


    for i in range(len(rules)):
        rule = rules[i]
        #calculate CUSTOM METRIC for rules: z-score for two sample test of equality of the mean in two binomial distributions
        ff1=clm.get_fourfold(i+1,1)
        ff2=clm.get_fourfold(i+1,2)
        p1 = ff1[0]/(ff1[0]+ff1[1])
        p2 = ff2[0]/(ff2[0]+ff2[1])
        n1= ff1[0]+ff1[1]
        n2= ff2[0]+ff2[1]
        p=(n1*p1+n2*p2)/(n1+n2)
        z=(p1-p2)/math.sqrt(p*(1-p)*(1/n1+1/n2))
        p_val=scipy.stats.norm.cdf(z)
        #calculate relative base to filter out rules that does not meet criteria on relative base on the full set
        relbase1 = ff1[0]/rows
        relbase2 = ff2[0]/rows
        print(f"Rule id {i+1}: Action rule : , p1={p1}, p2={p2}, z-value: {z:.2f}, p-value {p_val:.100f}, relbase1 {relbase1}, relbase2 {relbase2}")
        if relbase1>=relbase and relbase2>=relbase:
            cnt += 1
            d={}
            d['id']=cnt
            d['rule_id']=i+1
            d['z_score']=z
            newrules.append(d)
        else:
            print(f'...minimum relative base criterium not met. Relbase1 = {relbase1}, Relbase2 = {relbase2}')
            cnt_filtered_out+=1

    print(f"Total rules:{clm.get_rulecount()}, out of them action rules:{cnt+cnt_filtered_out}, for relative base criteria filtered out {cnt_filtered_out}, resulting rules {cnt}")
    ds['rules_act'][ind] = cnt


    import operator
    sorted_list = sorted(newrules, key=operator.itemgetter('z_score'))

    sorted_list.reverse()

    print(sorted_list)

    #TAKE TOP 30

    for i in range(min(30,cnt)):
        print(f"RULE {i+1}: z-score {sorted_list[i]['z_score']:.2f}, rule text: {clm.get_ruletext(sorted_list[i]['rule_id'])}")

    clm.print_rule(1)


print("SUMMARY OF ITERATIONS")
for itm in ds.index:
    print(f"Item {itm}: dataset name: {ds['filename'][itm]:<30}, rules total: {ds['rules_total'][itm]:>6}, action rules: {ds['rules_act'][itm]:>6}")


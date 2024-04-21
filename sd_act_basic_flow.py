import math

import pandas as pd
import sys
import sklearn.impute
#from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer

from cleverminer import cleverminer

import scipy

#exmaple that shows
# 1. Using CleverMiner SD4ft-Miner procedure to mine action rules with flexible antecedent
# - how to run task for Act4ft-Miner to generate rules via SD4ft Miner
# - how to filter out rules that does not meet ACT4ft Miner (same flexible part of antecedent)
# 2. (optional) How to use relative base requirement equivalent to action rules when using SD4ft-Miner
# - how to handle with relative base problem
# 3. improvements by CleverMiner not achievable by LISp-Miner
# - how to calculate own metrics (z-score)
# - how to sort rules by this metrics and show top N rules with highest value of the own metrics

df = pd.read_csv ('data/data_src/accidents.zip', encoding='cp1250', sep='\t')
df=df[['Driver_Age_Band','Driver_IMD','Sex','Area','Journey','Road_Type','Speed_limit','Light','Vehicle_Location','Vehicle_Type','Vehicle_Age','Hit_Objects_in','Hit_Objects_off','Casualties','Severity']]


imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)


def action_rules(df=None,quantifiers=None,ante=None,succ=None,ante_var=None):
    if (df is None or quantifiers is None or ante is None or succ is None or ante_var is None):
        print('Some attribute is missing. Please use all named attributes')
        exit(1)


    clm = cleverminer(df=df,proc='SD4ftMiner',quantifiers= quantifiers,
                      ante=ante,succ=succ,frst=ante_var, scnd=ante_var )
    return clm

clm = action_rules(df=df,
                  quantifiers={'RelBase1': 0.01, 'RelBase2': 0.01, 'Ratiopim': 1.2},
                  ante={
                      'attributes': [
                          {'name': 'Sex', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                      ], 'minlen': 1, 'maxlen': 4, 'type': 'con'},
                  succ={
                      'attributes': [
                          {'name': 'Severity', 'type': 'lcut', 'minlen': 1, 'maxlen': 2}
                      ], 'minlen': 1, 'maxlen': 1, 'type': 'con'},
                  ante_var={
                      'attributes': [
                          {'name': 'Journey', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                          {'name': 'Speed_limit', 'type': 'seq', 'minlen': 1, 'maxlen': 2}

                      ], 'minlen': 1, 'maxlen': 1, 'type': 'con'}
                  )


#print(clm.result)
#clm.print_rulelist()
#clm.print_summary()

rules = clm.result.get('rules')

#print(rules)

newrules=[]

cnt=0
cnt_filtered_out=0

rows= len(df)

for i in range(len(rules)):
    rule = rules[i]
    cedents_ids = rule.get('trace_cedent_taskorder')
    frst =cedents_ids.get('frst')
    scnd = cedents_ids.get('scnd')
#if attributes in first set and second set are the same (and only attribute value may change)- condition for action rules
    if (frst==scnd):
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
        if relbase1>=0.01 and relbase2>=0.01:
            cnt += 1
            d={}
            d['id']=cnt
            d['rule_id']=i+1
            d['z_score']=z
            newrules.append(d)
        else:
            print(f'...minimum relative base crite  rium not met. Relbase1 = {relbase1}, Relbase2 = {relbase2}')
            cnt_filtered_out+=1

print(f"Total rules:{clm.get_rulecount()}, out of them action rules:{cnt+cnt_filtered_out}, for relative base criteria filtered out {cnt_filtered_out}, resulting rules {cnt}")

import operator
sorted_list = sorted(newrules, key=operator.itemgetter('z_score'))

sorted_list.reverse()

print(sorted_list)

#TAKE TOP 30

for i in range(min(30,cnt)):
    print(f"RULE {i+1}: original rule {sorted_list[i]}")

clm.print_rule(16)
clm.print_rule(17)
clm.print_rule(24)
clm.print_rule(20)
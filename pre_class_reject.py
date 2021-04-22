# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:46:08 2021

@author: Administrator
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scorecard_class import ScoreCard

random_seed = 7
# from sklearn import tree
# import re

#%% 
#----------------------- 一. 导入数据 -----------------------#
All_Data = pd.read_csv("data_in_.csv")


#%%
#----------------------- 二. 数据预处理 -----------------------#
# *0. Drop D0&D1 Label
All_Data = All_Data.drop( All_Data[All_Data['cust_tag']=='D0'].index, axis=0 )
All_Data = All_Data.drop( All_Data[All_Data['cust_tag']=='D1'].index, axis=0 )

# 1. Drop CUS_NUM, label
All_Data = All_Data.drop(['CUS_NUM'], axis=1)
Y = All_Data['cust_tag']
All_Data = All_Data.drop(['cust_tag'], axis=1)
Y = Y.replace('E', 1)
Y = Y.replace('F', 1)
Y = Y.replace('A', 0)

# 2. Drop Features having "nan">95%;  KEY FUNC: df.count()
FEA_moreNan = list()
non_NA_All_Data = All_Data.count()
for i in non_NA_All_Data.index:
    if ( (len(All_Data)-non_NA_All_Data[i])/len(All_Data)>0.95 ):
        FEA_moreNan.append(i)
All_Data = All_Data.drop(FEA_moreNan, axis=1)

# 3. Drop Features having (same value)>95%; KEY FUNC: df['a'].value_counts()
FEA_moreSame = list()
for i in range(0, All_Data.shape[1]):
    non_nan_counts = All_Data.iloc[:,i].count()
    if ( any(All_Data.iloc[:,i].value_counts() > ( (All_Data.shape[0]-non_nan_counts)*0.95) ) ):
        FEA_moreSame.append(i)
FEA_moreSame = All_Data.columns[FEA_moreSame]
All_Data = All_Data.drop(FEA_moreSame, axis=1)        

# 4. Transfer nan to -99 And 'blank' Respectively
All_Data_types = pd.Series(All_Data.dtypes, dtype='str')

for i in range(0, np.shape(All_Data)[1]): # loop by columns
    # Check nan existence
    if any( All_Data.iloc[:, i].isna() ):
        if All_Data_types[i]=='float64' or All_Data_types[i]=='int64':
            nan_index = np.where(All_Data.iloc[:, i].isna())[0] #np.where() returns a tuple with only 1 element, just extract it.
            All_Data.iloc[nan_index, i] = -99
        else:
            nan_index = np.where(All_Data.iloc[:, i].isna())[0]
            All_Data.iloc[nan_index, i] = 'blank'

#%%
#----------------------- 三. 划分拒绝/放贷样本->对放贷样本划分测试集训练集合 -----------------------#
# 构造放贷/拒绝样本: 设置IK=3
# 放贷样本中 1：4000个，0：3000个
# 拒绝样本中 1：4000个，0：1000个
Good = All_Data.iloc[np.where(Y==1)[0]]
Bad = All_Data.iloc[np.where(Y==0)[0]]

Accept_Good = Good.sample(n=4000,axis=0,random_state=random_seed)
Accept_Bad = Bad.sample(n=3000,axis=0,random_state=random_seed)
Accept = pd.concat([Accept_Good,Accept_Bad], axis=0)
Accept_Y = Y.filter(Accept.index, axis=0)

Reject_Good_index = list(set(Good.index.tolist()) - set(Accept.index.tolist()))
Reject_Bad_index = list(set(Bad.index.tolist()) - set(Accept.index.tolist()))
Reject_Good_index = Good.filter(Reject_Good_index, axis=0)
Reject_Bad_index = Bad.filter(Reject_Bad_index, axis=0)

Reject = pd.concat([Reject_Good_index, Reject_Bad_index], axis=0)
Reject_Y = Y.filter(Reject.index, axis=0)

#重洗index
Accept.reset_index(drop=True,inplace=True)
Accept_Y.reset_index(drop=True,inplace=True)
Reject.reset_index(drop=True,inplace=True)
Reject_Y.reset_index(drop=True,inplace=True)

print('构造放贷样本中好坏样本比例：\n',Accept_Y.value_counts())
print('构造拒绝样本中好坏样本比例：\n',Reject_Y.value_counts())

x_train, x_test, y_train, y_test = train_test_split(Accept, Accept_Y,
                                                    test_size=0.3, random_state=random_seed)
X = x_train
Y = y_train

#%%

# 评分卡对象
sc = ScoreCard(X,Y)
sc_test = ScoreCard(x_test, y_test)
sc_reject = ScoreCard(Reject, Reject_Y)

# woe分箱
binning_return = sc.woe_tree(random_seed=random_seed)
print(binning_return['box_num_list'].apply(lambda x:np.sum(x)).unique())

# 用三个模型过滤特征
fea_models_return = sc.filter_feature_by_3_models(binning_return)
x_woe = fea_models_return['x']
y = fea_models_return['y']
Fea_choosed_en_name = fea_models_return['Fea_choosed_en_name']

x_test_woe = sc_test.orig_2_woe(binning_return)
x_reject_woe = sc_reject.orig_2_woe(binning_return)

# 用共线性过滤特征
corr_return = sc.filter_feature_by_correlation(x_woe,Fea_choosed_en_name,binning_return)
x = x_woe[corr_return['Fea_choosed_en_name']]

x_test_woe = x_test_woe[corr_return['Fea_choosed_en_name']]
x_reject_woe = x_reject_woe[corr_return['Fea_choosed_en_name']]

# lr建模
Lr_return = sc.lr(x,y)

# ks
print('-------训练集-------')
auc_ks_return = sc.auc_ks(Lr_return['model'], x, y)
print('-------测试集-------')
auc_ks_return_test = sc_test.auc_ks(Lr_return['model'], x_test_woe, y_test.reset_index(drop=True))
print('-------拒绝集-------')
auc_ks_return_reject = sc_reject.auc_ks(Lr_return['model'], x_reject_woe, Reject_Y)

#%%
'''
展开法-简单展开法（硬截断 hard-cutoff）：
step 1. 构建 KGB 模型，并对全量样本打分，得到 P(Good) 。
step 2. 将拒绝样本按 P(Good) 降序排列，设置 cutoff 。根据业务经验，
        比如拒绝样本的 bad rate 是放贷样本的2～4倍，从而结合拒绝样本量计算出 cutoff。
step 3. 高于 cutoff 的拒绝样本标记为 good ，反之标记为 bad 。
step 4. 利用组合样本构建 AGB 模型。
'''

def cutoff(ctff, auc_ks_return_reject):
    '''
    函数解释：
        给定一个ctff，返回在该cutoff下，给定预测结果的bad_rate
    '''
    cutoff = float(ctff)
    y_pred_reject = auc_ks_return_reject['y_pred']
    
    y_pred_reject = pd.DataFrame(y_pred_reject, columns=['prob_0', 'prob_1'])
    y_pred_reject = pd.concat([y_pred_reject, pd.DataFrame(np.zeros([y_pred_reject.shape[0],]), columns=['pred'])], axis=1)
    
    def get_result(x):
        if x>cutoff:
            return 1
        else:
            return 0
        
    y_pred_reject['pred'] = y_pred_reject['prob_1'].apply(lambda x:get_result(x))
    bad_rate = y_pred_reject['pred'].value_counts()[1]/y_pred_reject['pred'].value_counts()[0]
    
    result = {'y_pred_reject':y_pred_reject, 'bad_rate':bad_rate}
    
    return result


# cutoff=0.5时，bad_rate=2.4248455730954017

cutoff_range = np.arange(0.40,0.50,0.01)

min_err = np.inf
min_err_cutoff = 0
min_err_bad_rate = 0

for i in cutoff_range:    
    ctf_result = cutoff(i, auc_ks_return_reject)
    if np.abs(ctf_result['bad_rate']-3) < min_err:
        print(np.abs(ctf_result['bad_rate']-3))
        min_err = np.abs(ctf_result['bad_rate']-3)
        min_err_cutoff = i
        min_err_bad_rate = ctf_result['bad_rate']

Reject_Y_infer = cutoff(min_err_cutoff, auc_ks_return_reject)['y_pred_reject']['pred']























































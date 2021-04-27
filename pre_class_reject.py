# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:46:08 2021

@author: Administrator
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scorecard_class import ScoreCard
from tqdm import tqdm

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
#sc_test = ScoreCard(x_test, y_test)
#sc_reject = ScoreCard(Reject, Reject_Y)

# 转换类别型变量
sc_transform = sc.transform_discrete()
sc_test_transform = ScoreCard.transform_discrete_static(x_test, sc_transform)
sc_reject_transform = ScoreCard.transform_discrete_static(Reject, sc_transform)

# woe分箱
binning_return = ScoreCard.woe_tree(transform_discrete_return=sc_transform,random_seed=random_seed)
print(binning_return['box_num_list'].apply(lambda x:np.sum(x)).unique())

# binning_return = sc.woe_tree(random_seed=random_seed)
# print(binning_return['box_num_list'].apply(lambda x:np.sum(x)).unique())

# 用三个模型过滤特征
fea_models_return = ScoreCard.filter_feature_by_3_models(sc_transform,binning_return)
x_woe = fea_models_return['x']
y = fea_models_return['y']
Fea_choosed_en_name = fea_models_return['Fea_choosed_en_name']

x_test_woe = ScoreCard.orig_2_woe(sc_test_transform['x_new'], binning_return)
x_reject_woe = ScoreCard.orig_2_woe(sc_reject_transform['x_new'], binning_return)

# 用共线性过滤特征
corr_return = sc.filter_feature_by_correlation(x_woe,Fea_choosed_en_name,binning_return)
x = x_woe[corr_return['Fea_choosed_en_name']]

x_test_woe = x_test_woe[corr_return['Fea_choosed_en_name']]
x_reject_woe = x_reject_woe[corr_return['Fea_choosed_en_name']]

# lr建模
Lr_return = sc.lr(x,y)

# ks
print('-------训练集-------')
auc_ks_return = ScoreCard.auc_ks(Lr_return['model'], x, y)
print('-------测试集-------')
auc_ks_return_test = ScoreCard.auc_ks(Lr_return['model'], x_test_woe, y_test.reset_index(drop=True))
print('-------拒绝集(真实)-------')
auc_ks_return_reject = ScoreCard.auc_ks(Lr_return['model'], x_reject_woe, Reject_Y)

#%% 各种推断法

class InferReject:
    '''
        Accept_Data : DF
            第一列是Accept_Y
        Reject_Data : DF
            第一列是Reject_Y
        reject : dict
            auc_ks_return_reject
    '''
    def __init__(self, Accept_Data, Reject_Data, KGB_model):        
        self.Accept = Accept_Data
        self.Reject = Reject_Data
        self.Accept_Y = self.Accept.iloc[:,0]
        self.Reject_Y = self.Reject.iloc[:,0]
        self.Accept = self.Accept.iloc[:,1:]
        self.Reject = self.Reject.iloc[:,1:]
        
        # self.accept_data = Accept_Data
        # self.reject_data = Reject_Data
        self.kgb = KGB_model
        
    def hard_cutoff(self, ideal_bad_rate=3):
        '''
        ideal_bad_rate: 目标坏客户率，默认3
        展开法-简单展开法（硬截断 hard-cutoff）：
        step 1. 构建 KGB 模型，并对拒绝样本打分，得到 P(Good) 。
        step 2. 将拒绝样本按 P(Good) 降序排列，设置 cutoff 。根据业务经验，
                比如拒绝样本的 bad rate 是放贷样本的2～4倍，从而结合拒绝样本量计算出 cutoff。
        step 3. 高于 cutoff 的拒绝样本标记为 good ，反之标记为 bad 。
        step 4. 利用组合样本构建 AGB 模型。
        
        tips: 实际使用中，尤其是再bad样本很少的情况下，也可以只取出Reject预测分数低于20%的作为infer_bad，
              然后再建AGB。剩下高于20%的作为灰色样本，不予考虑
        return: performance_return
        '''
        # step 1: 得到KGB预测的返回
        Accept = self.Accept
        Reject = self.Reject
        Accept_Y = self.Accept_Y
        # Reject_Y先放在这儿，实际上没有用到，因为实际未知
        Reject_Y = self.Reject_Y
        kgb = self.kgb
        
        Reject_y_pred = kgb.predict_proba(Reject)
        
        # step 2: 求cutoff
        cutoff_range = np.arange(0.30,0.70,0.01)

        min_err = np.inf # 最小误差(某个cutoff下，预测的reject坏客户率与3,即Accept坏客户率的3倍,的差值)
        min_err_cutoff = 0 # 最小误差时的cutoff
        min_err_bad_rate = 0 # 最小误差时的坏客户率(非常接近3)
        
        for i in tqdm(cutoff_range):    
            ctf_result = InferReject.cutoff(i, Reject_y_pred)
            # 尽可能靠近目标bad_rate
            if np.abs(ctf_result['bad_rate']-ideal_bad_rate) < min_err:
                print(np.abs(ctf_result['bad_rate']-ideal_bad_rate))
                min_err = np.abs(ctf_result['bad_rate']-ideal_bad_rate)
                min_err_cutoff = i
                min_err_bad_rate = ctf_result['bad_rate']

        # step 3: cutoff函数执行替换y_pred为该cutoff下的label
        Reject_Y_infer = InferReject.cutoff(min_err_cutoff, Reject_y_pred)['y_pred_reject']['pred']

        # step 4: 重新组合样本后，构建AGB模型
        All_infer = pd.concat([Accept, Reject], axis=0).reset_index(drop=True)
        All_infer_Y = pd.concat([Accept_Y, Reject_Y_infer], axis=0).reset_index(drop=True)
        
        x_train_all_infer, x_test_all_infer, y_train_all_infer, y_test_all_infer = train_test_split(All_infer, All_infer_Y,
                                                            test_size=0.3, random_state=random_seed)
        
        # 评分卡对象
        sc_train_all_infer = ScoreCard(x_train_all_infer, y_train_all_infer)
        sc_test_all_infer = ScoreCard(x_test_all_infer, y_test_all_infer)
        
        performance_return = InferReject.get_performance(sc_train_all_infer, sc_test_all_infer)
                        
        return performance_return
       
    def fuzzy_augmentation(self):
        '''
        展开法-模糊展开法
        step 1. 构建 KGB 模型，并对拒绝样本打分，得到 P(Good) 和 P(Bad) 。
        step 2. 将每条拒绝样本复制为不同类别，不同权重的两条：
                一条标记为good ，权重为 P(Good) ；另一条标记为 bad ，权重为 P(Bad) 。
        step 3. 利用变换后的拒绝样本和放贷已知好坏样本（类别不变，权重设为1）建立 AGB 模型。
        '''
        # step 1: 得到KGB预测的返回
        Accept = self.Accept
        Reject = self.Reject
        Accept_Y = self.Accept_Y
        Reject_Y = self.Reject_Y
        kgb = self.kgb
        
        Reject_y_pred = kgb.predict_proba(Reject)
        
        # step 2: 
        y_pred = Reject_y_pred
        
        Reject_0 = Reject.copy()
        for i in tqdm(range(0, Reject_0.shape[0])):
            Reject_0.iloc[i,:] = Reject_0.iloc[i,:].apply(lambda x: x*y_pred[i,0])
        Reject_0_Y = pd.Series(np.zeros(Reject_0.shape[0]), name='cust_tag')
        
        Reject_1 = Reject.copy()
        for i in tqdm(range(0, Reject_1.shape[0])):
            Reject_1.iloc[i,:] = Reject_1.iloc[i,:].apply(lambda x: x*y_pred[i,1])
        Reject_1_Y = pd.Series(np.ones(Reject_0.shape[0]), name='cust_tag')
        
        Reject_new = pd.concat([Reject_0, Reject_1], axis=0)
        Reject_new_Y = pd.concat([Reject_0_Y, Reject_1_Y], axis=0)
        
        # step 3: 合并数据，建立AGB模型
        All = pd.concat([Accept, Reject_new], axis=0)
        All_Y = pd.concat([Accept_Y, Reject_new_Y], axis=0)
        
        x_train_all_infer, x_test_all_infer, y_train_all_infer, y_test_all_infer = \
                            train_test_split(All, All_Y, test_size=0.3, random_state=random_seed)
        
        # 评分卡对象
        sc_train_all_infer = ScoreCard(x_train_all_infer, y_train_all_infer)
        sc_test_all_infer = ScoreCard(x_test_all_infer, y_test_all_infer)
        
        performance_return = InferReject.get_performance(sc_train_all_infer, sc_test_all_infer)
        
        return performance_return
    
    def reweighting(self, box_num=5):
        '''
        重新加权法并没有把拒绝样本加入建模，只是调整了放贷好坏样本的权重。操作步骤为：

        step 1. 构建 KGB 模型，并对全量样本打分，得到 p(Good) 。
        step 2. 将全量样本按 p(Good) 降序排列，分箱统计每箱中的放贷和拒绝样本数。
        step 3. 计算每个分箱中放贷好坏样本的权重：
        Weight = (Reject_i+Accept_i)/Accept_i = (Reject_i+Good_i+Bad_i)/(Good_i+Bad_i)
        step 4. 引入样本权重，利用放贷好坏样本重新构建 KGB 模型。
        '''
        # step 1: 得到KGB的返回
        Accept = self.Accept
        Reject = self.Reject
        Accept_Y = self.Accept_Y
        Reject_Y = self.Reject_Y
        kgb = self.kgb
        
        All = pd.concat([Accept, Reject], axis=0).reset_index(drop=True)
        All_y_pred = kgb.predict_proba(All)[:,1]
        
        All_xy = pd.concat([pd.Series(All_y_pred),All], axis=1)
        Accept_xy = All_xy.iloc[:Accept.shape[0], :]
        Reject_xy = All_xy.iloc[Accept.shape[0]:, :].reset_index(drop=True)
        
        Accept_xy.insert(0,'weight',pd.Series(np.zeros([Accept_xy.shape[0]])))
        Reject_xy.insert(0,'weight',pd.Series(np.zeros([Reject_xy.shape[0]])))
        
        # step 2,3: 等频分箱: 第一列是权重，第二列是y
        step = 1/box_num
        records = pd.DataFrame(np.zeros([box_num,6]), columns=['min', 'max', 'total', 'accept', 'reject', 'weight'])
        
        for i in range(0, box_num):
            this_range_start = i*step
            this_range_end = (i+1)*step

            this_range_Accept_index = np.where( (Accept_xy.iloc[:,1]>this_range_start)&(Accept_xy.iloc[:,1]<=this_range_end) )[0]
            this_range_Reject_index = np.where( (Reject_xy.iloc[:,1]>this_range_start)&(Reject_xy.iloc[:,1]<=this_range_end) )[0]
            this_range_Accept = Accept_xy.iloc[this_range_Accept_index, :]            
            this_range_Reject = Reject_xy.iloc[this_range_Reject_index, :]            
            
            this_range_weight = (this_range_Reject.shape[0]+this_range_Accept.shape[0])/this_range_Accept.shape[0]
            
            # 写入权重
            Accept_xy.iloc[this_range_Accept_index, 0] = this_range_weight
            Reject_xy.iloc[this_range_Reject_index, 0] = this_range_weight
            
            # 写入records
            records.iloc[i,0] = this_range_start
            records.iloc[i,1] = this_range_end
            records.iloc[i,2] = this_range_Accept.shape[0]+this_range_Reject.shape[0]
            records.iloc[i,3] = this_range_Accept.shape[0]
            records.iloc[i,4] = this_range_Reject.shape[0]
            records.iloc[i,5] = this_range_weight
         
        # step 4: 提取权重，重新建立KGB        
        
        x_train_all_infer, x_test_all_infer, y_train_all_infer, y_test_all_infer = \
                            train_test_split(Accept_xy, Accept_Y, test_size=0.3, random_state=random_seed)
        x_train_all_infer_weight = x_train_all_infer.iloc[:,0]
        x_test_all_infer_weight = x_test_all_infer.iloc[:,0]
        # drop掉weight，y
        x_train_all_infer.drop(0,axis=1,inplace=True)
        x_train_all_infer.drop('weight',axis=1,inplace=True)
        x_test_all_infer.drop(0,axis=1,inplace=True)
        x_test_all_infer.drop('weight',axis=1,inplace=True)
        
        # 评分卡对象
        sc_train_all_infer = ScoreCard(x_train_all_infer, y_train_all_infer)
        sc_test_all_infer = ScoreCard(x_test_all_infer, y_test_all_infer)
        
        performance_return = InferReject.get_performance(sc_train_all_infer, sc_test_all_infer, sample_weight=x_train_all_infer_weight)
 
        return performance_return
    
    '''
    以下为通用函数
    '''
    @staticmethod
    def cutoff(ctff, y_pred):
        '''
        函数解释：
            给定一个float(ctff)和一个预测返回的y_pred，
            返回在该cutoff下，给定预测结果，和 bad_rate
        '''
        cutoff = float(ctff)
        y_pred_reject = y_pred.copy()
        
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
    
    @staticmethod
    def get_performance(train_card, test_card, **kw):
        '''
        函数解释：
            给定train, test的评分卡对象，以及可选择的建模参数，输出预测表现，返回两个的auc_ks_return
        '''
        
        # woe分箱
        transform_discrete_return = train_card.transform_discrete()
        binning_return_infer = train_card.woe_tree(transform_discrete_return, random_seed=random_seed)
        
        # 用三个模型过滤特征
        fea_models_return_infer = ScoreCard.filter_feature_by_3_models(transform_discrete_return, binning_return_infer)
        x_woe_infer = fea_models_return_infer['x']
        y_infer = fea_models_return_infer['y']
        Fea_choosed_en_name_infer = fea_models_return_infer['Fea_choosed_en_name']
        
        x_test_woe_infer = ScoreCard.orig_2_woe(test_card.orig_x, binning_return_infer)
        
        # 用共线性过滤特征
        corr_return_infer = ScoreCard.filter_feature_by_correlation(x_woe_infer,Fea_choosed_en_name_infer,binning_return_infer)
        x_infer = x_woe_infer[corr_return_infer['Fea_choosed_en_name']]
        
        x_test_woe_infer = x_test_woe_infer[corr_return_infer['Fea_choosed_en_name']]
        
        # lr建模
        # if len(kw)==0:
        #     Lr_return_infer = ScoreCard.lr(x_infer,y_infer)
        # else:
        #     Lr_return_infer = ScoreCard.lr(x_infer,y_infer,kw=kw)
        Lr_return_infer = ScoreCard.lr(x_infer,y_infer,kw=kw)
        
        # ks
        print('-------训练集-------')
        auc_ks_return_infer = ScoreCard.auc_ks(Lr_return_infer['model'], x_infer, y_infer)
        print('-------测试集-------')
        auc_ks_return_test_infer = ScoreCard.auc_ks(Lr_return_infer['model'], x_test_woe_infer, test_card.orig_y.reset_index(drop=True))
    
        performance_return = {'Lr_return':Lr_return_infer,
                              'auc_ks_return_infer': auc_ks_return_infer,
                              'auc_ks_return_test_infer':auc_ks_return_test_infer}
        
        return performance_return


# 取得woe替换的原数据
sc_Accept_transform = ScoreCard.transform_discrete_static(Accept, sc_transform)

Reject_woe = x_reject_woe.copy()
Accept_woe = ScoreCard.orig_2_woe(sc_Accept_transform['x_new'], binning_return)
Accept_woe = Accept_woe.loc[:, Reject_woe.columns]

kgb = Lr_return['model']
ir = InferReject(pd.concat([Accept_Y,Accept_woe],axis=1), pd.concat([Reject_Y,Reject_woe],axis=1), kgb)

hard_cutoff_performance = ir.hard_cutoff()
fuzzy_augmentation_performance = ir.fuzzy_augmentation()
reweighting_performance = ir.reweighting()



        





















































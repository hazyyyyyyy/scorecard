# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:33:39 2021

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import statsmodels.api as sm
import re
import math

class ScoreCard:
    
    '''
    以下是实例函数，用于转换原数据中的类别变量为数值变量
    '''
    def __init__(self, orig_x, orig_y):
        self.orig_x = orig_x
        self.orig_y = orig_y
    
    def transform_discrete(self): #按照bad_rate转换类别型变量为数值
        x = self.orig_x.copy()
        y = self.orig_y.copy()
        x.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        x_types = pd.Series(x.dtypes, dtype='str')
        
        x_num_position = np.where( (x_types=='float64')|(x_types=='int64') )[0]
        x_num = x.iloc[:, x_num_position]
        #x_num_types = x_types.iloc[x_num_position]   
        x_category_position = np.where( x_types=='object' )[0]
        x_category = x.iloc[:, x_category_position]
        #x_category_types = x_types.iloc[x_category_position] # All 'object'
        
        # 记录了每个变量的数值转换规则
        x_category_transfer_rule = list(np.zeros([len(x_category.columns)])) 
        # 对应的变量名称
        x_category_transfer_rule_featurename = pd.Series(list(range(0,len(x_category.columns))), index=x_category.columns)
        # 类别变量转化后的值
        x_category_transform = pd.DataFrame(np.zeros(x_category.shape), columns=x_category.columns)
        
        for i in range(0, x_category.shape[1]):
            unique_value = np.unique(x_category.iloc[:,i])
            unique_value = unique_value[np.where(unique_value!='blank')]
            this_transform_rule = pd.concat([pd.DataFrame(unique_value,columns=['raw_data']),\
                                            pd.DataFrame(np.zeros([len(unique_value),2]), columns=['transform_data', 'bad_rate'])],
                                            axis=1)
            
            for j in range(0, len(unique_value)):
                this_unique = unique_value[j]
                this_bad_num = len(np.where( (y==1)&(x_category.iloc[:,i]==this_unique) )[0])
                this_good_num = len(np.where( (y==0)&(x_category.iloc[:,i]==this_unique) )[0])
                # 防止good_num为零报错
                if this_good_num==0:
                    this_good_num=0.5
                this_transform_rule.iloc[j,2] = this_bad_num/this_good_num
            
            this_transform_rule = this_transform_rule.sort_values(by='bad_rate')
            this_transform_rule.iloc[:,1] = list(range(len(unique_value),0,-1))
            this_transform_rule.reset_index(drop=True, inplace=True)
            # 填回nan值
            if len(unique_value)!=len(x_category.iloc[:, i].unique()):
                this_transform_rule = this_transform_rule.append( pd.DataFrame({'raw_data': ['blank'], 'transform_data': [-99], 'bad_rate': [np.nan]}) )
            # 规则装回总表
            x_category_transfer_rule[i] = this_transform_rule
            
            # 把整个转换后的值按规则填回原数据表
            unique_value = x_category.iloc[:, i].unique()
            for j in range(0, len(unique_value)):
                x_category_transform.iloc[np.where(x_category.iloc[:, i]==unique_value[j])[0], i] = this_transform_rule.iloc[np.where(this_transform_rule.iloc[:, 0]==unique_value[j])[0], 1].iloc[0]
        # 合并连续型变量
        x_category_transform.reset_index(drop=True, inplace=True)
        x_num.reset_index(drop=True, inplace=True)
        x_new = pd.concat([x_category_transform, x_num], axis=1)
        
        rules = {'x_category_transfer_rule':x_category_transfer_rule,
                 'x_category_transfer_rule_featurename':x_category_transfer_rule_featurename}
        transform_discrete_return = {'x_new':x_new,
                                     'y':y,
                                     'x_types':x_types,
                                     'rules':rules}
        
        return transform_discrete_return

    '''
    以下是通用静态函数
    '''
    
    @staticmethod
    def transform_discrete_static(x_orig, transform_discrete_return):
        '''
        x_orig: 原始x(注意static方法不要求y)
        transform_discrete_return: 给定一个transform_discrete的返回，根据里面的rules转换x_orig
        return: 返回转换后的x_new
        '''
        x = x_orig.copy()
        # 这是一个df
        x_category_transfer_rule = transform_discrete_return['rules']['x_category_transfer_rule']
        # 这是一个Series
        x_category_transfer_rule_featurename = transform_discrete_return['rules']['x_category_transfer_rule_featurename']
        
        x.reset_index(drop=True, inplace=True)
        x_category = x.loc[:,x_category_transfer_rule_featurename.index]
        # 类别变量转换后的值
        x_category_transform = x_category.copy()
        x_num = x.drop(x_category_transfer_rule_featurename.index, axis=1)
        
        for this_fea in x_category_transfer_rule_featurename.index:
            i = x_category_transfer_rule_featurename[this_fea]
            this_rule = x_category_transfer_rule[i]
            this_rule.reset_index(drop=True, inplace=True)
            this_fea_unique = x_category_transform[this_fea].unique()
            
            # 检验unique值是否是规则的子集，如果不是，则直接结束函数
            if not set(this_fea_unique).issubset( set(x_category_transfer_rule[i]['raw_data']) ):
                print('unique值非给定规则子集,特征名：',this_fea)
                return
            # 检验结束后，开始替换原值
            for j in range(0, this_rule.shape[0]):
                this_unique = this_rule.iloc[j,0]
                this_unique_transform = this_rule.iloc[j,1]
                x_category_transform[this_fea].iloc[ np.where(x_category_transform[this_fea]==this_unique)[0] ] = this_unique_transform
        
        x_category_transform = x_category_transform.astype('int')
        x_new = pd.concat([x_category_transform, x_num], axis=1)         
        
        transform_discrete_static_return = {'x_new':x_new}
        return transform_discrete_static_return
    
    @staticmethod
    def cal_woe(data, dim, bucket_num=10, auto=True, bond_num=[]):
        m = data.shape[0] 
        X = data[:, dim]
        y = data[:, -1]
        tot_bad = np.sum(y == 1)
        tot_good = np.sum(y == 0)
        data = np.column_stack((X.reshape(m, 1), y.reshape(m, 1)))
        cnt_bad = []
        cnt_good = []
        min = np.min(data[:, 0])
        max = np.max(data[:, 0])
        # bucket is the space between 2 bondaries
        # index is the boundaries
        if auto == True:
            index = np.linspace(min, max, bucket_num + 1)
        else:
            index = bond_num
            bucket_num = bond_num.shape[0] - 1
        data_bad = data[data[:, 1] == 1, 0]  # Extract rows with y==1, and delete the y col
        data_good = data[data[:, 1] == 0, 0] # Extract rows with y==0, and delete the y col
        for i in range(bucket_num): # count bad/good_num per bucket
            cnt_bad.append(1.0 * np.sum(np.bitwise_and(data_bad > index[i], data_bad <= index[i + 1])))
            cnt_good.append(1.0 * np.sum(np.bitwise_and(data_good > index[i], data_good <= index[i + 1])))
        bond = np.array(index)
        cnt_bad = np.array(cnt_bad)
        cnt_good = np.array(cnt_good)
        
        #对完美分箱增加一个虚拟样本，保证有woe值
        cnt_bad[cnt_bad==0]+=1
        cnt_good[cnt_good==0]+=1
        
        
        length = cnt_bad.shape[0]
        for i in range(length): # 检验分箱内是否只有一种y(向前合并)
            j = length - i - 1
            ## after combing, i refers to the elements after proevious single-y's element,
            ## so the cnt list would contain no single-y bucket after one single loop
            if j != 0:
                # this bucket has only one kind of 'y' -> combine with former
                if cnt_bad[j] == 0 or cnt_good[j] == 0:
                    cnt_bad[j - 1] += cnt_bad[j]
                    cnt_good[j - 1] += cnt_good[j]
                    cnt_bad = np.append(cnt_bad[:j], cnt_bad[j + 1:])
                    cnt_good = np.append(cnt_good[:j], cnt_good[j + 1:])
                    bond = np.append(bond[:j], bond[j + 1:])
        ## since we only combine with former one bucket,
        ## necessary to check the 1st bucket
        if cnt_bad[0] == 0 or cnt_good[0] == 0:
            cnt_bad[1] += cnt_bad[0]
            cnt_good[1] += cnt_good[0]
            cnt_bad = cnt_bad[1:]
            cnt_good = cnt_good[1:]
            bond = np.append(bond[0], bond[2:])
        woe = np.log((cnt_bad / tot_bad) / (cnt_good / tot_good))
        IV = ((cnt_bad / tot_bad) - (cnt_good / tot_good)) * woe
        IV_tot = np.sum(IV)
        bond_str = []
        for b in bond:
            bond_str.append(str(b))
        box_num=  cnt_bad+ cnt_good
        bad_rate=cnt_bad/box_num
        
        return IV_tot, IV, woe, bond, box_num, bad_rate

    @staticmethod
    def woe_tree(transform_discrete_return, min_elem_per_box_ratio=0.05, max_box_num=5, tolerance_despite_nan=0, random_seed=None):
        '''
            min_elem_per_box_ratio: 最小每箱数据条数的比例
            max_box_num: 最多箱数
            tolerance_despite_nan: 除了nan最多允许的拐点数量
            return: binning_return字典
        '''
        x = transform_discrete_return['x_new']
        y = transform_discrete_return['y']
        x_types = transform_discrete_return['x_types']
        
        data = pd.concat([x,y], axis=1)
        x_split_points = list(np.zeros([len(x.columns), 1]))
        
        # cal_woe的返回
        woe_list = []
        IV_list=[]
        IV_tot_list=[]
        box_num_list=[]
        bad_rate_list=[]

        # 逐特征进行WOE分箱
        for i in range(0, x.shape[1]):
            # 最小每箱数据条数的比例
            min_woe_boxing_num = int(round(len(y)*min_elem_per_box_ratio))
            
            # 'nan' 作独立分箱
            thisFea_nonnan_position = np.where(x.iloc[:, i]!=-99)[0]
            
            # 如果存在nan，设置最多箱数-1
            if len(thisFea_nonnan_position) == x.shape[0]:
                max_bin_num = max_box_num
            else:
                max_bin_num = max_box_num-1           
            
            # 开始分箱
            continue_bining = True
            while continue_bining:
                this_x = x.iloc[thisFea_nonnan_position, i]
                this_x = this_x.values.reshape(-1, 1) # -1 means every row
                this_y = y.iloc[thisFea_nonnan_position]
                this_y = this_y.values.reshape(-1, 1)
                
                # 单列决策树拟合
                groupdt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=min_woe_boxing_num, max_leaf_nodes=max_bin_num, random_state=random_seed)
                groupdt.fit(this_x, this_y)
                
                # 抽取决策树分箱后的数据
                dot_data = tree.export_graphviz(groupdt)
                pattern = re.compile('<= (.*?)\\\\nentropy', re.S)
                split_num = re.findall(pattern, dot_data)
                split_point = [float(j) for j in split_num]
                final_split_points = sorted(split_point)

                # 在分割点两端填充-inf和inf
                final_split_points.insert(0, -np.inf)
                final_split_points.append(np.inf)
                if len(thisFea_nonnan_position) != x.shape[0]: # 'nan' Exists
                    final_split_points.insert(1, -99) # 使用别的数据集时必须确定-99是最小的值
                
                # 完成当前分箱后，计算WOE，并检查其单调性，
                # 如果不单调，max_leaf_nodes-1后重跑
                IV_tot, IV, woe, bond, box_num, bad_rate = ScoreCard.cal_woe(np.array(data), i, auto='False', bond_num=pd.Series(final_split_points))

                if x_types.loc[x.columns[i]]=='object' or len(woe)<=3:
                    break # 如果是类别型变量，或箱数不高于3，直接跳过下面的矫正
                
                # after obtaining the WOE table,
                # changing-direction points should be checked
                woe_direction = np.sign(woe[1:] - woe[:-1])
                # direction_change_points == 2 or -2 Means Direction Changes
                direction_change_points = np.abs(woe_direction[1:]-woe_direction[:-1])
                direction_chang_times = np.sum( direction_change_points==2 )
                
                # Now, if the nan at the 1st place results to the np.abs(direction_change_points[0])==2
                # Then direction_change_times -= 1
                if len(thisFea_nonnan_position) != len(x) and direction_change_points[0] == 2:
                    direction_chang_times -= 1
                
                if direction_chang_times == 0: # Bining Succeed
                    continue_bining = False
                else: # WOE not monotone, re-bining
                    max_bin_num -= 1
                    continue_bining = True

            x_split_points[i]=final_split_points   
            woe_list.append(woe)
            IV_list.append(IV)
            IV_tot_list.append(IV_tot)
            box_num_list.append(box_num)
            bad_rate_list.append(bad_rate)
        
        ################## WOE Bining DOOOOOOOOOOOOOOOOOOOOONE! ##################
        x_split_points = pd.Series(x_split_points, index=x.columns)
        woe_list = pd.Series(woe_list, index=x.columns)
        IV_list = pd.Series(IV_list, index=x.columns)
        IV_tot_list = pd.Series(IV_tot_list,index=x.columns)
        box_num_list = pd.Series(box_num_list,index=x.columns)
        bad_rate_list = pd.Series(bad_rate_list,index=x.columns)
        
        binning_return = {'x_split_points':x_split_points,\
                          'woe_list':woe_list,\
                          'IV_list':IV_list,\
                          'IV_tot_list':IV_tot_list,\
                          'box_num_list':box_num_list,\
                          'bad_rate_list':bad_rate_list}

        return binning_return

    @staticmethod
    def orig_2_woe(x_new, binning_return):
        x_split_points = binning_return['x_split_points']
        woe_list = binning_return['woe_list']
        x = x_new.copy()
        
        # x = transform_discrete_return['x_new']
        # y = transform_discrete_return['y']
        # x_types = transform_discrete_return['x_types']
        
        # 1. Transform Datasets into WOE bin
        for thisFea in range(0, len(x.columns)):
            thisFea_split = x_split_points[thisFea]
            thisFea_woe = woe_list[thisFea]
            for j in range(0, len(thisFea_split)-1): # loop stop at 2nd last element
                x_thisFea_thisBox = np.where( (x.iloc[:, thisFea]>thisFea_split[j]) & (x.iloc[:, thisFea]<=thisFea_split[j+1]) )[0]
                x.iloc[x_thisFea_thisBox, thisFea] = thisFea_woe[j]
        return x
        
    @staticmethod
    def filter_feature_by_3_models(transform_discrete_return, binning_return,choose_2=True,Lasso_threshold = 0.01,RF_threshold = 0.001,IV_threshold = 0.1):
        '''
        binning_return:woe分箱的返回字典
        choose_2:True-被至少两个模型选中的变量；False-被三个模型均选中多个变量
        Lasso_threshold:LASSO模型的阈值
        RF_threshold:随机森林特征重要性的阈值
        IV_threshold:IV的阈值
        return: fea_models_return字典。其中x为woe替换原值数据,Fea_choosed_en_name为选中的特征
        
        函数解释：
        转换数据原值为对应区间的woe值，
        然后用三个模型根据阈值进行筛选
        '''
        
        #x_split_points = binning_return['x_split_points']
        #woe_list = binning_return['woe_list']
        IV_tot_list = binning_return['IV_tot_list']
        
        x_new = transform_discrete_return['x_new']
        y = transform_discrete_return['y']
        x_types = transform_discrete_return['x_types']
        
        # 1. Transform Datasets into WOE bin
        '''
        for thisFea in range(0, len(x.columns)):
            thisFea_split = x_split_points[thisFea]
            thisFea_woe = woe_list[thisFea]
            for j in range(0, len(thisFea_split)-1): # loop stop at 2nd last element
                x_thisFea_thisBox = np.where( (x.iloc[:, thisFea]>thisFea_split[j]) & (x.iloc[:, thisFea]<=thisFea_split[j+1]) )[0]
                x.iloc[x_thisFea_thisBox, thisFea] = thisFea_woe[j]
        '''
        x = ScoreCard.orig_2_woe(x_new, binning_return)
        
        # 2. Calculate Feature Importance by LASSO
        x_scaled = preprocessing.scale(x)    
        Logit_Lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.3)
        Logit_Lasso.fit(x_scaled, y)
        coef_by_Lasso = np.abs(Logit_Lasso.coef_).T
        coef_by_Lasso = pd.Series(coef_by_Lasso.reshape(-1), index = x.columns)
        
        
        # 3. Calculate Feature Importance by Random Forest
        RF_clf = RandomForestClassifier(n_estimators=1000, criterion='gini', n_jobs=-1, max_depth=10, min_samples_leaf=10)
        RF_clf.fit(x, y)
        coef_by_RF = RF_clf.feature_importances_
        #coef_by_RF = coef_by_RF.reshape(-1, 1) # in accordance with the dimentionality of coef_by_Lasso
        coef_by_RF = pd.Series(coef_by_RF.reshape(-1), index = x.columns)
        
        
        # 4. Calculate Feature Importance by IV
        coef_by_IV = IV_tot_list.copy()
        
        # 5. Select Features by Feature Importance of LASSO&RF       
        Fea_choosed_by_Lasso = coef_by_Lasso.copy()
        Fea_choosed_by_Lasso[ np.where(Fea_choosed_by_Lasso>Lasso_threshold)[0] ] = 1
        Fea_choosed_by_Lasso[ np.where(Fea_choosed_by_Lasso<=Lasso_threshold)[0] ] = 0
        
        Fea_choosed_by_RF = coef_by_RF.copy()
        Fea_choosed_by_RF[ np.where(Fea_choosed_by_RF>RF_threshold)[0] ] = 1
        Fea_choosed_by_RF[ np.where(Fea_choosed_by_RF<=RF_threshold)[0] ] = 0
        
        Fea_choosed_by_IV = coef_by_IV.copy()
        Fea_choosed_by_IV[ np.where(Fea_choosed_by_IV>IV_threshold)[0] ] = 1
        Fea_choosed_by_IV[ np.where(Fea_choosed_by_IV<=IV_threshold)[0] ] = 0
        
        Fea_choosed = Fea_choosed_by_Lasso + Fea_choosed_by_RF + Fea_choosed_by_IV       
        if choose_2:
            feature_choosed_by_models = np.where(Fea_choosed >= 2)[0]
        else:
            feature_choosed_by_models = np.where(Fea_choosed == 3)[0]
        Fea_choosed_en_name = list(x.columns[feature_choosed_by_models])
        fea_models_return={'x':x,
                           'y':y,
                           'Fea_choosed_en_name':Fea_choosed_en_name,
                           'Fea_choosed_by_IV':Fea_choosed_by_IV,
                           'Fea_choosed_by_Lasso':Fea_choosed_by_Lasso,
                           'Fea_choosed_by_RF':Fea_choosed_by_RF}

        return fea_models_return
    
    @staticmethod
    def filter_feature_by_correlation(x_woe,Fea_choosed,binning_return,corrcoef_threshold = 0.4,VIF_threshold = 10):
        '''
        x_woe: woe替换后的数据（filter_feature_by_3_models的第一份返回值）
        Fea_choosed: 选中的特征（filter_feature_by_3_models的第二份返回值）
        binning_return: woe分箱的返回字典
        corrcoef_threshold: Pearson阈值
        VIF_threshold  VIF阈值
        return: corr_return字典。其中Fea_choosed_by_pearson为第一步pearson结果
        '''
        x = x_woe.copy()
        Fea_choosed_en_name = Fea_choosed.copy()
        IV_tot_list = binning_return['IV_tot_list']
        # 1. 先根据Pearson筛选变量
        corrcoef_matrix = np.corrcoef(x.loc[:, Fea_choosed_en_name].T)
        for i in range(0, len(corrcoef_matrix)): # 对角线设为零
            corrcoef_matrix[i, i] = 0
        high_corrcoef = np.where( corrcoef_matrix >= corrcoef_threshold )
        
        while len(high_corrcoef[0])>0:
            first_Fea = Fea_choosed_en_name[ high_corrcoef[0][0] ]
            second_Fea = Fea_choosed_en_name[ high_corrcoef[1][0] ]
            
            if (IV_tot_list.loc[first_Fea] >= IV_tot_list.loc[second_Fea]):
                Fea_choosed_en_name.remove(second_Fea)
            else:
                Fea_choosed_en_name.remove(first_Fea)
            # Recompute the corrcoef_matrix
            corrcoef_matrix = np.corrcoef(x.loc[:, Fea_choosed_en_name].T)
            for i in range(0, len(corrcoef_matrix)): # 对角线设为零
                corrcoef_matrix[i, i] = 0
            high_corrcoef = np.where( corrcoef_matrix >= corrcoef_threshold )
        Fea_choosed_by_pearson = Fea_choosed_en_name.copy()
        
        # 2. 再根据VIF筛选变量
        vif = list()
        for i in range(0, len(Fea_choosed_en_name)):
            this_VIF = variance_inflation_factor(np.array(x.loc[:, Fea_choosed_en_name]), i)
            vif.append(this_VIF)
                
        vif = pd.Series(vif, index = Fea_choosed_en_name)
        
        keep_VIF = np.where(vif<VIF_threshold)[0]
        Fea_choosed_en_name = np.array(Fea_choosed_en_name)[keep_VIF].tolist()
        corr_return = {'Fea_choosed_by_pearson':Fea_choosed_by_pearson,
                       'corrcoef_matrix':corrcoef_matrix,
                       'Fea_choosed_en_name':Fea_choosed_en_name}
        return corr_return

    @staticmethod
    def rfe():
        '''
        空函数，还没做
        '''
        return
    
    @staticmethod
    def stepwise_ks():
        '''
        空函数，还没做
        '''
        return
    
    @staticmethod
    def lr(x, y, default=True):
        '''
        x: x_woe
        y: y
        default: 默认参数：penalty='l2', c=0.3
        return: Lr_return字典。其中model为sklearn的Lr模型对象
        '''
        if default:
            Lr= LogisticRegression(penalty='l2',C=0.3)
            Lr.fit(x, y) 
            
            coef_Lr=Lr.coef_.transpose()  #提取特征权重
            coef_intercept=Lr.intercept_  #提取截距
            
            # p test
            logit = sm.Logit(y, x)
            result = logit.fit()
            print('Parameters: ', result.params)
            margeff = result.get_margeff(dummy=True)
            print(margeff.summary())
            
            Lr_return = {'model': Lr,
                         'coef_Lr': coef_Lr,
                         'coef_intercept': coef_intercept}
            return Lr_return
        else:
            return {}
        
    @staticmethod
    def auc_ks(Lr_model, x, y_true):
        y_pred = Lr_model.predict_proba(x)
        
        y_0=list(y_pred[:,1])
        fpr,tpr,thresholds=roc_curve(y_true,y_0)  #计算fpr,tpr,thresholds
        auc=roc_auc_score(y_true,y_0) #计算auc
        print('auc为',auc)
        
        #画曲线图
        plt.figure()
        plt.plot(fpr,tpr)
        plt.title('$ROC curve$')
        plt.show()
         
        #计算ks
        KS_max=0
        best_thr=0
        for i in range(len(fpr)):
            if(i==0):
                KS_max=tpr[i]-fpr[i]
                best_thr=thresholds[i]
            elif (tpr[i]-fpr[i]>KS_max):
                KS_max = tpr[i] - fpr[i]
                best_thr = thresholds[i]
        
        print('最大KS为：',KS_max)
        print('最佳阈值为：',best_thr)
        auc_ks_return = {'auc':auc,
                         'ks':KS_max,
                         'y_pred':y_pred,
                         'fpr':fpr,
                         'tpr':tpr}
        
        return auc_ks_return
            
    
    # def create_scorecard(self,x, y, point0 = 600,odds0 = 0.05,PDO = 40):
    #     '''
    #     参数含义： odds0=bad_rate/good_rate=0.05时，point0=600；odds*2时，point变化值为PDO=40
    #     '''
    #     # 根据给定基准值，求参数A, B以及初始分数score0
    #     # 公式：Point = A - B * log(odds)
    #     B = PDO/math.log(2)
    #     A = point0+B*math.log(odds0)
    #     score0=A-B*coef_intercept #初始分数
        
    #     # 初始化评分卡
    #     chosen_feature_num = x.shape[1]
    #     ScoreCard = list(np.zeros([chosen_feature_num,]))
    #     # 变量每一分箱区间的分数 = -该段woe值*该变量的逻辑回归系数*评分卡参数B
        
    #     for i in range(0, chosen_feature_num):
    #         this_Fea = chosen_feature_final[i]
    #         this_FeaType = x_train_types.loc[this_Fea]
    #         this_Coef = coef_Lr[i]
    #         this_SplitPoint = x_split_points.loc[this_Fea]
    #         this_WOE = woe_list.loc[this_Fea]
    #         this_SplitNum = len(this_SplitPoint)-1
            
    #         this_ScoreCard = pd.DataFrame(np.zeros([this_SplitNum, 2]))
            
    #         if this_FeaType=='float64' or this_FeaType=='int64':
    #             for j in range(0, this_SplitNum):
    #                 if j == this_SplitNum-1:
    #                     this_ScoreCard.iloc[j,0]='('+str(this_SplitPoint[j])+','+str(this_SplitPoint[j+1])+')'
    #                 else:
    #                     this_ScoreCard.iloc[j,0]='('+str(this_SplitPoint[j])+','+str(this_SplitPoint[j+1])+']'
    #                 this_ScoreCard.iloc[j,1] = -this_WOE[j]*this_Coef*B
                
    #         else:
    #             this_TranRule = x_train_category_tranRules[ x_train_category_tranRules_feaName.loc[this_Fea] ]
                
    #             for j in range(0, this_SplitNum):
    #                 this_Bin_Raw = this_TranRule.iloc[ np.where((this_TranRule.loc[:, 'Transform data']>this_SplitPoint[j])&(this_TranRule.loc[:, 'Transform data']<=this_SplitPoint[j+1]))[0], 0 ]
                    
    #                 this_ScoreCard.iloc[j, 0] = this_Bin_Raw.iloc[0]
    #                 for k in range(1, len(this_Bin_Raw)):
    #                     this_ScoreCard.iloc[j,0] = this_ScoreCard.iloc[j,0] + ', ' + this_Bin_Raw.iloc[k]
                    
    #                 this_ScoreCard.iloc[j, 1] = -this_WOE[j]*this_Coef*B
        
    #         ScoreCard[i] = this_ScoreCard
    























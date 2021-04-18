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
import re

class ScoreCard:
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
        return x_new, y, x_types

    def cal_woe(self, data, dim, bucket_num=10, auto=True, bond_num=[]):
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

    def woe_tree(self, min_elem_per_box_ratio=0.05, max_box_num=5, tolerance_despite_nan=0):
        '''
            min_elem_per_box_ratio: 最小每箱数据条数的比例
            max_box_num: 最多箱数
            tolerance_despite_nan: 除了nan最多允许的拐点数量
            return: binning_return字典
        '''
        x, y, x_types = self.transform_discrete()
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
                groupdt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=min_woe_boxing_num, max_leaf_nodes=max_bin_num)
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
                IV_tot, IV, woe, bond, box_num, bad_rate = self.cal_woe(np.array(data), i, auto='False', bond_num=pd.Series(final_split_points))

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

    def filter_feature_by_3_models(self, binning_return,choose_2=True,Lasso_threshold = 0.01,RF_threshold = 0.001,IV_threshold = 0.1):
        '''
        binning_return:woe分箱的返回字典
        choose_2:True-被至少两个模型选中的变量；False-被三个模型均选中多个变量
        Lasso_threshold:LASSO模型的阈值
        RF_threshold:随机森林特征重要性的阈值
        IV_threshold:IV的阈值
        return: x-woe替换后的数据,Fea_choosed_en_name-选中的特征
        
        函数解释：
        转换数据原值为对应区间的woe值，
        然后用三个模型根据阈值进行筛选
        '''
        x_split_points = binning_return['x_split_points']
        woe_list = binning_return['woe_list']
        IV_tot_list = binning_return['IV_tot_list']
        
        x, y, x_types = self.transform_discrete()
        # 1. Transform Datasets into WOE bin
        for thisFea in range(0, len(x.columns)):
            thisFea_split = x_split_points[thisFea]
            thisFea_woe = woe_list[thisFea]
            for j in range(0, len(thisFea_split)-1): # loop stop at 2nd last element
                x_thisFea_thisBox = np.where( (x.iloc[:, thisFea]>thisFea_split[j]) & (x.iloc[:, thisFea]<=thisFea_split[j+1]) )[0]
                x.iloc[x_thisFea_thisBox, thisFea] = thisFea_woe[j]
                                
        # 2. Calculate Feature Importance by LASSO
        x_scaled = preprocessing.scale(x)    
        Logit_Lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.3, n_jobs=-1)
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

        return x, Fea_choosed_en_name
    
    def filter_feature_by_correlation(self,x_woe,Fea_choosed_en_name,binning_return,corrcoef_threshold = 0.4,VIF_threshold = 10):
        '''
        x_woe : woe替换后的数据（filter_feature_by_3_models的第一份返回值）
        Fea_choosed_en_name : 选中的特征（filter_feature_by_3_models的第二份返回值）
        binning_return : woe分箱的返回字典
        corrcoef_threshold : Pearson阈值
        VIF_threshold : VIF阈值
        return : Fea_choosed_en_name经过共线性筛选后的特征
        '''
        x = x_woe.copy()
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
                
        # 2. 再根据VIF筛选变量
        vif = list()
        for i in range(0, len(Fea_choosed_en_name)):
            this_VIF = variance_inflation_factor(np.array(x.loc[:, Fea_choosed_en_name]), i)
            vif.append(this_VIF)
                
        vif = pd.Series(vif, index = Fea_choosed_en_name)
        
        high_VIF = np.where(vif>VIF_threshold)[0]
        for i in range(0, len(high_VIF)):
            Fea_choosed_en_name.remove( Fea_choosed_en_name[high_VIF[i]] )

        return Fea_choosed_en_name





































































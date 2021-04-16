# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:33:39 2021

@author: Administrator
"""

import pandas as pd
import numpy as np

class ScoreCard:
    def __init__(self, orig_x, orig_y):
        self.orig_x = orig_x
        self.orig_y = orig_y
    def transform_discrete(self): #按照transform rule的bad_rate转换类别型变量为数值
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
                                            pd.DataFrame(np.zeros([len(unique_value),2]), columns=['transform_data', 'bad_rate'])])
            
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
        
    


































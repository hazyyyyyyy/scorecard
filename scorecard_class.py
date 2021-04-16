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
        x = self.orig_x
        y = self.orig_y
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
        #      
        
        
        
    


































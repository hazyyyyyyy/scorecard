import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.feature_selection import RFE,RFECV
from sklearn.metrics import roc_curve,auc,confusion_matrix,accuracy_score,cohen_kappa_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math

def WOE(data, dim, bucket_num=10, auto=True, bond_num=[]):
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
    eps = 1e-8
    for i in range(bucket_num): # count bad/good_num per bucket
        if i < bucket_num - 1: # before  i==(bucket_num-1)
            cnt_bad.append(1.0 * np.sum(np.bitwise_and(data_bad > index[i] - eps, data_bad < index[i + 1])))
            cnt_good.append(1.0 * np.sum(np.bitwise_and(data_good > index[i] - eps, data_good < index[i + 1])))
        else:
            cnt_bad.append(1.0 * np.sum(np.bitwise_and(data_bad > index[i] - eps, data_bad < index[i + 1] + eps)))
            cnt_good.append(1.0 * np.sum(np.bitwise_and(data_good > index[i] - eps, data_good < index[i + 1] + eps)))
    bond = np.array(index)
    cnt_bad = np.array(cnt_bad)
    cnt_good = np.array(cnt_good)
    
    #对完美分箱增加一个虚拟样本，保证有woe值
    cnt_bad[cnt_bad==0]+=1
    cnt_good[cnt_good==0]+=1
    
    
    length = cnt_bad.shape[0]
    for i in range(length):
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

def cal_ks(point,Y,section_num=20):  
    Y=pd.Series(Y)
    sample_num=len(Y)
    
    bad_percent=np.zeros([section_num,1])
    good_percent=np.zeros([section_num,1])

    
    point=pd.DataFrame(point)
    sorted_point=point.sort_values(by=0)
    total_bad_num=len(np.where(Y==1)[0])
    total_good_num=len(np.where(Y==0)[0])
    
    for i in range(0,section_num):
        split_point=sorted_point.iloc[int(round(sample_num*(i+1)/section_num))-1]
        position_in_this_section=np.where(point<=split_point)[0]
        bad_percent[i]=len(np.where(Y.iloc[position_in_this_section]==1)[0])/total_bad_num
        good_percent[i]=len(np.where(Y.iloc[position_in_this_section]==0)[0])/total_good_num
        
    ks_value=bad_percent-good_percent

    return ks_value,bad_percent,good_percent

def PSI(score_train,score_test,section_num=10):
    score_train=pd.DataFrame(score_train)
    score_test=pd.DataFrame(score_test)
    
    total_train_num=len(score_train)
    total_test_num=len(score_test)
    
    sorted_score_train=score_train.sort_values(by=0)    
    
    PSI_value=0
    
    for i in range(0,section_num):
        lower_bound=sorted_score_train.iloc[int(round(total_train_num*(i)/section_num))]
        higher_bound=sorted_score_train.iloc[int(round(total_train_num*(i+1)/section_num))-1]
        score_train_percent=len(np.where((score_train>=lower_bound)&(score_train<=higher_bound))[0])/total_train_num
        score_test_percent=len(np.where((score_test>=lower_bound)&(score_test<=higher_bound))[0])/total_test_num
        
        PSI_value+=(score_test_percent-score_train_percent)*np.log(score_test_percent/score_train_percent)
        
    return PSI_value
    
#%% 
#----------------------- 一. 导入数据 -----------------------#
All_Data = pd.read_csv("data_in.csv")

#%%
#----------------------- 二. 数据预处理 -----------------------#
# *0. Drop D0&D1 Label
All_Data = All_Data.drop( All_Data[All_Data['cust_tag']=='D0'].index, axis=0 )
All_Data = All_Data.drop( All_Data[All_Data['cust_tag']=='D1'].index, axis=0 )

# 1. Drop CUS_NUM
All_Data = All_Data.drop(['CUS_NUM'], axis=1)

# 2. Drop Features having "nan">95%;  KEY FUNC: df.count()
FEA_moreNan = list()
non_NA_All_Data = All_Data.count()
for i in non_NA_All_Data.index:
    if ( (len(All_Data)-non_NA_All_Data[i])/len(All_Data)>0.95 ):
        FEA_moreNan.append(i)
All_Data = All_Data.drop(FEA_moreNan, axis=1)

# 3. Drop Features having (same value)>95%; KEY FUNC: df['a'].value_counts()
FEA_moreSame = list()
for i in All_Data:
    if ( any(All_Data[i].value_counts() > ( len(All_Data)*0.95) ) ):
        FEA_moreSame.append(i)
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
    
'''
X_types=pd.Series(X.dtypes,dtype='str')

for i in range(0,np.shape(X)[1]):
    if len(np.where(X.iloc[:,i].isna())[0])>0: #若有缺失值
        if X_types[i]=='float64' or X_types[i]=='int64':#若为数值型，则填充为-99           
            X.iloc[np.where(X.iloc[:,i].isna())[0],i]=-99            
            #判断原数据是否是整数型，后面SMOTE也会用到
            if len(np.where(X.iloc[np.where(~X.iloc[:,i].isna())[0],i]-np.array(X.iloc[np.where(~X.iloc[:,i].isna())[0],i],dtype='int'))[0])==0:
                X_types[i]='int64'          
        else:#若为分类型，则填充为blank
            X.iloc[np.where(X.iloc[:,i].isna())[0],i]='blank'
'''

#%%
#----------------------- 三. 划分测试集训练集合 -----------------------#
x_train, x_test, y_train, y_test = train_test_split(All_Data.iloc[:, 1:], All_Data.iloc[:, 0],
                                                    test_size=0.3, random_state=7)

#%%
#----------------------- 四. 处理样本不均衡问题 -----------------------#
'''
'''

#%%
#----------------------- 五. 样本WOE分箱 -----------------------#
# 1. Divide Categorical and Continuous Features
## train
x_train_types = pd.Series(x_train.dtypes, dtype='str')

x_train_num_position = np.where( (x_train_types=='float64')|(x_train_types=='int64') )[0]
x_train_num = x_train.iloc[:, x_train_num_position]
x_train_num_types = x_train_types.iloc[x_train_num_position]

x_train_category_position = np.where( x_train_types=='object' )[0]
x_train_category = x_train.iloc[:, x_train_category_position]
x_train_category_types = x_train_types.iloc[x_train_category_position] # All 'object'

## test
x_test_types = pd.Series( x_test.dtypes, dtype='str' )

x_test_num_position = np.where( (x_test_types=='float64')|(x_test_types=='int64') )[0]
x_test_num = x_test.iloc[:, x_test_num_position]
x_test_num_types = x_test_types.iloc[x_test_num_position]

x_test_category_position = np.where( x_test_types=='object' )[0]
x_test_category = x_test.iloc[:, x_test_category_position]
x_test_category_types = x_test_types.iloc[x_test_category_position] # All 'object'

# 2. Establish Feature Transfer Rules Table
x_train_category_tranRules = list(np.zeros(x_train_category.shape[1]))
x_train_category_tranRules_feaName = pd.Series( list(range(0, len(x_train_category.columns))), index=x_train_category.columns )
# 'x_train_category_afterTran' Is to save the transformed x_train_category
x_train_category_afterTran = pd.DataFrame(np.zeros(x_train_category.shape), columns=x_train_category.columns)

# Reset Index for the 2nd 'for j...'Loop Below
x_train_category_reindex = x_train_category.copy()
x_train_category_reindex.index = list(range(0, len(x_train_category)))
x_train_category_afterTran.index = x_train_category_reindex.index

# By Constructig pd.DataFrame Like This, the 'index' is Already Resetted
x_test_category_afterTran = pd.DataFrame(np.zeros(x_test_category.shape), columns=x_test_category.columns)
x_test_category_reindex = x_test_category.copy()
x_test_category_reindex.index = list(range(0, len(x_test_category)))

# Let 'F', 'E', as BAD(1), 'A' as 'GOOD'(0) -> Transform y_train
y_train_bin = y_train.copy()
y_train_bin = y_train_bin.replace(['F', 'E'], 1)
y_train_bin = y_train_bin.replace('A', 0)

y_train_bin_reindex = y_train_bin.copy()
y_train_bin_reindex.index = list(range(0, len(y_train_bin_reindex)))

y_test_bin = y_test.copy()
y_test_bin = y_test_bin.replace(['F', 'E'], 1)
y_test_bin = y_test_bin.replace('A', 0)

y_test_bin_reindex = y_test_bin.copy()
y_test_bin_reindex.index = list(range(0, len(y_test_bin_reindex)))

for i in range(0, len(x_train_category_reindex.columns)):
    # Current Feature's Unique Number
    unique_value = x_train_category_reindex.iloc[:, i].unique()
    # Ignore 'blank' For Now
    unique_value = unique_value[ np.where(unique_value!='blank') ]
    # Establish current Transfer Rule Table
    this_transfer_rule = pd.DataFrame( columns=['Raw data', 'Transform data', 'Bad rate'])
    this_transfer_rule['Raw data'] = unique_value
    
    for j in range(0, len(unique_value)):
        # Let 'F', 'E', as BAD, 'A' as 'GOOD'
        ij_bad_num = len( np.where((x_train_category_reindex.iloc[:, i]==unique_value[j])&(np.array(y_train_bin)==1))[0] )
        ij_good_num = len( np.where((x_train_category_reindex.iloc[:, i]==unique_value[j])&(np.array(y_train_bin)==0))[0] )
        #print('ij_bad_num: ', ij_bad_num)
        #print('ij_good_num:' , ij_good_num)
        # In Case ij_good_num==0 which would cause error of dividing 0
        if ij_good_num == 0:
            ij_good_num = 0.5
        ij_bad_rate = ij_bad_num/ij_good_num       
        # Fill In 'Bad rate'
        this_transfer_rule.iloc[j, 2] = ij_bad_rate
        
    this_transfer_rule = this_transfer_rule.sort_values(by='Bad rate', ascending=True)
    this_transfer_rule.iloc[:, 1] = list(range(len(this_transfer_rule), 0, -1))
    # Now Deal with Nan
    if len(unique_value)!=len(x_train_category_reindex.iloc[:, i].unique()):
        this_transfer_rule = this_transfer_rule.append( pd.DataFrame({'Raw data': ['blank'], 'Transform data': [-99], 'Bad rate': [np.nan]}) )
    # Fill in the bigger List
    x_train_category_tranRules[i] = this_transfer_rule
    
    # Fill 'blank' back in unique_value
    unique_value = x_train_category_reindex.iloc[:, i].unique()
    for j in range(0, len(unique_value)):
        x_train_category_afterTran.iloc[np.where(x_train_category_reindex.iloc[:, i]==unique_value[j])[0], i] = this_transfer_rule.iloc[np.where(this_transfer_rule.iloc[:, 0]==unique_value[j])[0], 1].iloc[0]
        x_test_category_afterTran.iloc[np.where(x_test_category_reindex.iloc[:, i]==unique_value[j])[0], i] = this_transfer_rule.iloc[np.where(this_transfer_rule.iloc[:, 0]==unique_value[j])[0], 1].iloc[0]

# 3. Establishing WOE bin data

# Construct the full training dataset after transformed
x_train_num_reindex = x_train_num.copy()
x_train_num_reindex.index = list(range(0, len(x_train_num_reindex)))
x_train_afterTran = pd.concat([x_train_num_reindex, x_train_category_afterTran], axis=1)

x_test_num_reindex = x_test_num.copy()
x_test_num_reindex.index = list(range(0, len(x_test_num_reindex)))
x_test_afterTran = pd.concat([x_test_num_reindex, x_test_category_afterTran], axis=1)

train_afterTran = pd.concat([x_train_afterTran, y_train_bin_reindex], axis=1)

# Split Points of Each Feature
x_split_points = list(np.zeros([len(x_train_afterTran.columns), 1])) # by specifying [len(...), 1], x_split_points is a list containing numpy array

# The return list of WOE Function
woe_list = []
IV_list=[]
IV_tot_list=[]
box_num_list=[]
bad_rate_list=[]

# WOE bining by each feature:
for i in range(0, len(x_train_afterTran.columns)):
    
    # Minimum Number of Elements of One Box
    min_woe_boxing_num = int(round(len(y_train_bin_reindex)*0.1))
    
    # 'nan' Should Be an Independent Box
    thisFea_nonnan_position = np.where(x_train_afterTran.iloc[:, i]!=-99)[0]
    
    # If 'nan' Exists, Max Bin Num = 4 (as nan will be the 5th)
    # Otherwise, Max Bin Num = 5
    if len(thisFea_nonnan_position) == len(x_train_afterTran):
        max_bin_num = 5
    else:
        max_bin_num = 4
    
    # Start Bining
    continue_bining = True
    while continue_bining:
        this_x = x_train_afterTran.iloc[thisFea_nonnan_position, i]
        this_x = this_x.values.reshape(-1, 1) # -1 means every row
        this_y = y_train_bin_reindex.iloc[thisFea_nonnan_position]
        this_y = this_y.values.reshape(-1, 1)
        
        # Fit the Decision Tree
        groupdt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=min_woe_boxing_num, max_leaf_nodes=max_bin_num)
        groupdt.fit(this_x, this_y)
        
        # Extract Split Data
        dot_data = tree.export_graphviz(groupdt)
        pattern = re.compile('<= (.*?)\\\\nentropy', re.S)
        split_num = re.findall(pattern, dot_data)
        split_point = [float(j) for j in split_num]
        final_split_points = sorted(split_point)
        
        # Fill -inf, inf, and -99(The minimum of the Dataset is -99, which has been checked by using min() and np.array(df).flatten())
        final_split_points.insert(0, -np.inf)
        final_split_points.append(np.inf)
        if len(thisFea_nonnan_position) != len(x_train_afterTran): # 'nan' Exists
            final_split_points.insert(1, -99)
        
        # Done Current Bining,
        # Calculated WOE and then Check its Monotonicity(if not monotone, then max_bin_num-1 and re-bining)
        IV_tot, IV, woe, bond, box_num, bad_rate = WOE(np.array(train_afterTran), i, auto='False', bond_num=pd.Series(final_split_points))
        
        if x_train_types.loc[x_train_afterTran.columns[i]]=='object' or len(woe)<=3:
            break # This break the while Loop
        
        # after obtaining the WOE table,
        # changing-direction points should be checked
        woe_direction = np.sign(woe[1:] - woe[:-1])
        # direction_change_points == 2 or -2 Means Direction Changes
        direction_change_points = np.abs(woe_direction[1:]-woe_direction[:-1])
        direction_chang_times = np.sum( direction_change_points==2 )
        
        # Now, if the nan at the 1st place results to the np.abs(direction_change_points[0])==2
        # Then direction_change_times -= 1
        if len(thisFea_nonnan_position) != len(x_train_afterTran) and direction_change_points[0] == 2:
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
x_split_points = pd.Series(x_split_points, index=x_train_afterTran.columns)
woe_list = pd.Series(woe_list, index=x_train_afterTran.columns)
IV_list = pd.Series(IV_list, index=x_train_afterTran.columns)
IV_tot_list = pd.Series(IV_tot_list,index=x_train_afterTran.columns)
box_num_list = pd.Series(box_num_list,index=x_train_afterTran.columns)
bad_rate_list = pd.Series(bad_rate_list,index=x_train_afterTran.columns)



#%%        
#----------------------- 六. 利用模型筛选变量 -----------------------#
# 方案一： 结合LASSO, 随机森林，IV值利用哑变量进行筛选
'''
# 1. Construct new x_train_num_dummies datasets with Num and Dummy Features

# Get Dummy Features of Category Features
x_train_category_dummies = pd.get_dummies(x_train_category_reindex)
x_train_category_dummies_reindex = x_train_category_dummies.copy()
x_train_category_dummies_reindex.index = list(range(0, len(x_train_category_dummies_reindex)))
# Map Dummy Features to its Original Feature
x_train_dummies_map = pd.Series(index = x_train_category_dummies_reindex.columns)
for this_dummy_name in x_train_dummies_map.index:
    for this_cate_name in x_train_category_reindex.columns:
        if this_dummy_name.find(this_cate_name)>=0:
            x_train_dummies_map.loc[this_dummy_name] = this_cate_name

# Concat x_train_num_reindex and x_train_category_dummies_reindex
x_train_num_dummies = pd.concat([x_train_num_reindex, x_train_category_dummies_reindex], axis=1)

# 2. Calculate Feature Importance by LASSO
Logit_Lasso = LogisticRegression(penalty='l1', solver='liblinear', n_jobs=-1)
Logit_Lasso.fit(x_train_num_dummies, y_train_bin_reindex)
coef_by_Lasso = np.abs(Logit_Lasso.coef_).T

# 3. Calculate Feature Importance by Random Forest
RF_clf = RandomForestClassifier(n_estimators=1000, criterion='gini', n_jobs=-1)
RF_clf.fit(x_train_num_dummies, y_train_bin_reindex)
coef_by_RF = RF_clf.feature_importances_
coef_by_RF = coef_by_RF.reshape(-1, 1) # in accordance with the dimentionality of coef_by_Lasso

# 4. Select Features by Feature Importance of LASSO&RF
Lasso_threshold = 0.02
RF_threshold = 0.01
IV_threshold = 0.05

# Lasso
num_choosed_by_Lasso = pd.Series(data = np.zeros(len(x_train_num_reindex.columns)), index = x_train_num_reindex.columns)
cate_choosed_by_Lasso = pd.Series(data = np.zeros(len(x_train_category_reindex.columns)), index = x_train_category_reindex.columns)
for i in range(0, len(x_train_num_reindex.columns)):
    if coef_by_Lasso[i] > Lasso_threshold: # Features Selected
        num_choosed_by_Lasso[i] = 1
for i in range(len(x_train_num_reindex.columns), len(coef_by_Lasso)):
    if coef_by_Lasso[i] > Lasso_threshold:
        this_cate_FEA = x_train_dummies_map.loc[ str(x_train_category_dummies_reindex.columns[i-len(x_train_num_reindex.columns)]) ]
        cate_choosed_by_Lasso[this_cate_FEA] = 1
Fea_choosed_by_Lasso = num_choosed_by_Lasso.append(cate_choosed_by_Lasso)

# Random Forest
num_choosed_by_RF = pd.Series(data = np.zeros(len(x_train_num_reindex.columns)), index = x_train_num_reindex.columns)
cate_choosed_by_RF = pd.Series(data = np.zeros(len(x_train_category_reindex.columns)), index = x_train_category_reindex.columns)
for i in range(0, len(x_train_num_reindex.columns)):
    if coef_by_RF[i] > RF_threshold: # Features Selected
        num_choosed_by_RF[i] = 1
for i in range(len(x_train_num_reindex.columns), len(coef_by_RF)):
    if coef_by_RF[i] > RF_threshold:
        this_cate_FEA = x_train_dummies_map.loc[ str(x_train_category_dummies_reindex.columns[i-len(x_train_num_reindex.columns)]) ]
        cate_choosed_by_RF[this_cate_FEA] = 1
Fea_choosed_by_RF = num_choosed_by_RF.append(cate_choosed_by_RF)

# IV
# num_choosed_by_IV = pd.Series(data = np.zeros(len(x_train_num_reindex.columns)), index = x_train_num_reindex.columns)
# cate_choosed_by_IV = pd.Series(data = np.zeros(len(x_train_category_reindex.columns)), index = x_train_category_reindex.columns)
Fea_choosed_by_IV = pd.Series( data = np.zeros(len(x_train.columns)), index = Fea_choosed_by_Lasso.index )
for this_Fea in IV_tot_list.index:
    if IV_tot_list[this_Fea] > IV_threshold: # Features Selected
      Fea_choosed_by_IV.loc[this_Fea] = 1
      
# Summary Features selected chose by these 3 models
Fea_choosed = Fea_choosed_by_Lasso + Fea_choosed_by_RF + Fea_choosed_by_IV
Fea_choosed_by_three = np.where(Fea_choosed == 3)[0]  ## only 2 features are chose by all 3 models...
Fea_choosed_by_two = np.where(Fea_choosed == 2)[0]
'''

# 方案二： 结合LASSO, 随机森林，IV值利用woe转化后的值进行筛选

# 1. Transform Datasets into WOE bin
x_train_woe_bin = x_train_afterTran.copy()
x_test_woe_bin = x_test_afterTran.copy()
for thisFea in range(0, len(x_train_woe_bin.columns)):
    thisFea_split = x_split_points[thisFea]
    thisFea_woe = woe_list[thisFea]
    for j in range(0, len(thisFea_split)-1): # loop stop at 2nd last element
        x_train_thisFea_thisBox = np.where( (x_train_woe_bin.iloc[:, thisFea]>thisFea_split[j]) & (x_train_woe_bin.iloc[:, thisFea]<=thisFea_split[j+1]) )[0]
        x_train_woe_bin.iloc[x_train_thisFea_thisBox, thisFea] = thisFea_woe[j]
        
        x_test_thisFea_thisBox = np.where( (x_test_woe_bin.iloc[:, thisFea]>thisFea_split[j]) & (x_test_woe_bin.iloc[:, thisFea]<=thisFea_split[j+1]) )[0]
        x_test_woe_bin.iloc[x_test_thisFea_thisBox, thisFea] = thisFea_woe[j]
        
# 2. Calculate Feature Importance by LASSO
x_train_woe_bin_scaled = preprocessing.scale(x_train_woe_bin)    
Logit_Lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.3, n_jobs=-1)
Logit_Lasso.fit(x_train_woe_bin_scaled, y_train_bin_reindex)
coef_by_Lasso = np.abs(Logit_Lasso.coef_).T
coef_by_Lasso = pd.Series(coef_by_Lasso.reshape(-1), index = x_train_woe_bin.columns)


# 3. Calculate Feature Importance by Random Forest
RF_clf = RandomForestClassifier(n_estimators=1000, criterion='gini', n_jobs=-1, max_depth=10, min_samples_leaf=10)
RF_clf.fit(x_train_woe_bin, y_train_bin_reindex)
coef_by_RF = RF_clf.feature_importances_
#coef_by_RF = coef_by_RF.reshape(-1, 1) # in accordance with the dimentionality of coef_by_Lasso
coef_by_RF = pd.Series(coef_by_RF.reshape(-1), index = x_train_woe_bin.columns)


# 4. Calculate Feature Importance by IV
coef_by_IV = IV_tot_list.copy()

# 5. Select Features by Feature Importance of LASSO&RF
Lasso_threshold = 0.01
RF_threshold = 0.001
IV_threshold = 0.1

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
Fea_choosed_by_three = np.where(Fea_choosed == 3)[0]  
Fea_choosed_by_two = np.where(Fea_choosed >= 2)[0]
Fea_choosed_en_name = list(x_train_woe_bin.columns[Fea_choosed_by_two])

#%%
#----------------------- 七. 利用共线性消除变量 -----------------------#
# 1. 先根据Pearson筛选变量
corrcoef_threshold = 0.6

corrcoef_matrix = np.corrcoef(x_train_woe_bin.loc[:, Fea_choosed_en_name].T)
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
    corrcoef_matrix = np.corrcoef(x_train_woe_bin.loc[:, Fea_choosed_en_name].T)
    for i in range(0, len(corrcoef_matrix)): # 对角线设为零
        corrcoef_matrix[i, i] = 0
    high_corrcoef = np.where( corrcoef_matrix >= corrcoef_threshold )

# 2. 再根据VIF筛选变量
VIF_threshold = 3
vif = list()

for i in range(0, len(Fea_choosed_en_name)):
    this_VIF = variance_inflation_factor(np.array(x_train_woe_bin.loc[:, Fea_choosed_en_name]), i)
    vif.append(this_VIF)
        
vif = pd.Series(vif, index = Fea_choosed_en_name)

high_VIF = np.where(vif>VIF_threshold)[0]
for i in range(0, len(high_VIF)):
    Fea_choosed_en_name.remove( Fea_choosed_en_name[high_VIF[i]] )
    
#%%    
#----------------------- 八. 利用循环特征(RFE)筛选变量 -----------------------#
# 1. Calculate RFE Ranking
x_train_selected_data = x_train_woe_bin.loc[:, Fea_choosed_en_name]
# Scale It to Avoid WOE Outliers
x_train_selected_data_scaled = pd.DataFrame(preprocessing.scale(x_train_selected_data), columns=x_train_selected_data.columns)

Lr1 = LogisticRegression(penalty='l2')
rfe = RFE(Lr1, n_features_to_select=1)
rfe.fit(x_train_selected_data_scaled, y_train_bin_reindex)

# Obtain Ranking of Feature importance to l2 lr model
rfe_ranking = rfe.ranking_
rfe_ranking = pd.Series(rfe_ranking, index=Fea_choosed_en_name)

# 2. Calculate BIC list for each feature combination
# BIC=k*ln(n)-2ln(L) 其中 k是特征数量,n是样本数量,L是模型似然值, 得到BIC值序列
subs_chosen_feature_num=list(range(5,len(rfe_ranking))) # 特征可取数目范围
chosen_feature_num_BIC=pd.Series(np.zeros([len(subs_chosen_feature_num),]),index=subs_chosen_feature_num)
# Only Calculate Features Ranking 5 to 15 (Adding to the Model One by One)
for i in range(subs_chosen_feature_num[0],subs_chosen_feature_num[-1]+1): # i = [5, 15)
    chosen_feature = list(x_train_selected_data_scaled.columns[np.where(rfe_ranking<=i)[0]])
    chosen_feature_data_final = x_train_selected_data.loc[:,chosen_feature]
    Lr= LogisticRegression(penalty='l2')
    Lr.fit(chosen_feature_data_final, y_train_bin_reindex)
    predict_prob_val = Lr.predict_proba(chosen_feature_data_final)
    chosen_feature_num_BIC.loc[i]=-2*sum(np.log(predict_prob_val[y_train_bin_reindex==1,1]))-2*sum(np.log(predict_prob_val[y_train_bin_reindex==0,0]))+i*np.log(len(x_train_selected_data.index))
'''
plt.figure()
plt.plot(chosen_feature_num_BIC)
plt.ylabel('BIC value')
plt.xlabel('number of features')
'''
feature_num_corres_min_BIC=subs_chosen_feature_num[int(np.where(chosen_feature_num_BIC==min(chosen_feature_num_BIC))[0])]

#%%
#----------------------- 九. 统计变量表现 -----------------------#
# 1. 观察每个变量分箱后的分布及其对应的坏客户率，若不合理，调整变量分箱点并更新woe转换值。

# 1.1 Plot Bad Rate Figure
## feature_num_corrs_min_BIC == 13
chosen_feature_num = 13 # All 22
chosen_feature_final = x_train_selected_data.columns[ np.where(rfe_ranking<=chosen_feature_num)[0] ]

observe_var = chosen_feature_final[7] # The variable you want to plot

this_bad_rate = bad_rate_list[observe_var]
this_box_num = box_num_list[observe_var]
this_splitpoint = x_split_points[observe_var]
this_IV_tot = IV_tot_list[observe_var]

this_section=[]
for i in range(0,len(this_splitpoint)-1):
    this_section.append('['+str(this_splitpoint[i])+','+str(this_splitpoint[i+1])+']')


'''画出占比图和坏客户率图'''
# fig,ax1 = plt.subplots()  
# ax2 = ax1.twinx() 
# ax1.bar(list(range(0,len(this_box_num))),this_box_num/np.sum(this_box_num), color='c')  
# ax1.set_xlabel('boxing section',color='r') 
# ax1.set_ylabel('samples percent')
# plt.xticks(list(range(0,len(this_box_num))),this_section)
# ax2.plot(this_bad_rate)
# ax2.set_ylabel('bad rate',color='b')  

# 1.2. Delete Certain Split Point if Needed
## This part is actually never gonna be needed,
## as the binning step makes sure only features with monotonical WOE would be kept
'''
this_splitpoint_temp = this_splitpoint.copy()
this_splitpoint_temp.remove(-99) # The value you wanna remove

# Calculate New WOE of This Variable
data = pd.concat([x_train_afterTran, y_train_bin_reindex], axis=1) # Parameter for WOE function
dim = np.where(x_train_afterTran.columns==observe_var)[0]
IV_tot, IV, woe, bond, box_num, bad_rate = WOE(np.array(data), dim, auto='False',bond_num=pd.Series(this_splitpoint_temp))

# Recheck the Bad Rate Plot
this_section_temp=[]
for i in range(0,len(this_splitpoint_temp)-1):
    this_section_temp.append('['+str(this_splitpoint_temp[i])+','+str(this_splitpoint_temp[i+1])+']')


# Deleting -99 of this specific feature actuall results to two turning-direction points
# 画出删除节点后的占比图和坏客户率图
fig,ax1 = plt.subplots()  
ax2 = ax1.twinx() 
ax1.bar(list(range(0,len(box_num))),box_num/np.sum(box_num), color='c')  
ax1.set_xlabel('boxing section',color='r') 
ax1.set_ylabel('samples percent')
plt.xticks(list(range(0,len(box_num))),this_section_temp)
ax2.plot(bad_rate)
ax2.set_ylabel('bad rate',color='b')  
'''

## 1.3. Replace Former Data with New Splits
'''
IV_tot_list[observe_var] = IV_tot
IV_list[observe_var] = IV
woe_list[observe_var] = woe
box_num_list[observe_var] = box_num
bad_rate_list[observe_var] = bad_rate

x_split_points.loc[observe_var] = this_splitpoint_temp

# Now Update WOE DATA: 'x_train_woe_bin'
thisFea = np.where(x_train_afterTran.columns==observe_var)[0][0] # This case, thisFea=143


for j in range(0, len(this_splitpoint_temp)-1): # loop stop at 2nd last element
    x_train_thisBox = np.where( (x_train_afterTran.iloc[:, thisFea]>this_splitpoint_temp[j]) & (x_train_afterTran.iloc[:, thisFea]<=this_splitpoint_temp[j+1]) )[0]
    x_train_woe_bin.iloc[x_train_thisBox, thisFea] = woe[j]
    
    x_test_thisBox = np.where( (x_test_afterTran.iloc[:, thisFea]>this_splitpoint_temp[j]) & (x_test_afterTran.iloc[:, thisFea]<=this_splitpoint_temp[j+1]) )[0]
    x_test_woe_bin.iloc[x_test_thisBox, thisFea] = woe[j]
'''
# 2. 计算选定变量之间的相关系数、VIF。
corrcoef_matrix_chosen_feature_final = np.corrcoef( x_train_woe_bin.loc[:,chosen_feature_final].T )

vif_final = list()

for i in range(0, len(chosen_feature_final)):
    this_VIF = variance_inflation_factor(np.array(x_train_woe_bin.loc[:, chosen_feature_final]), i)
    vif_final.append(this_VIF)
        
vif_final = pd.Series(vif_final, index = chosen_feature_final)


 
#%%
#----------------------- 十. 逻辑回归 -----------------------#

# 1. Logistic Regression
chosen_feature_data_final = x_train_woe_bin.loc[:, chosen_feature_final]

Lr= LogisticRegression(penalty='l2',C=0.3)
Lr.fit(chosen_feature_data_final, y_train_bin_reindex) 

coef_Lr=Lr.coef_.transpose()  #提取特征权重
coef_intercept=Lr.intercept_  #提取截距

# p test
logit = sm.Logit(y_train_bin_reindex, chosen_feature_data_final)
result = logit.fit()
print('Parameters: ', result.params)
margeff = result.get_margeff(dummy=True)
print(margeff.summary()) ## p('ac_min16_debit_in')=0.275


#%%
#----------------------- 十一. 建立评分卡 -----------------------#

# 根据给定基准值: point=600, when odds0=0.05; PDO=40
# 求参数A, B以及初始分数score0
# 公式：Point = A - B * log(odds)

point0 = 600
odds0 = 0.05
PDO = 40

B = PDO/math.log(2)
A = point0+B*math.log(odds0)
score0=A-B*coef_intercept #初始分数

# Initialize Score Card
ScoreCard = list(np.zeros([chosen_feature_num,]))
# 变量每一分箱区间的分数 = -该段woe值*该变量的逻辑回归系数*评分卡参数B

for i in range(0, chosen_feature_num):
    this_Fea = chosen_feature_final[i]
    this_FeaType = x_train_types.loc[this_Fea]
    this_Coef = coef_Lr[i]
    this_SplitPoint = x_split_points.loc[this_Fea]
    this_WOE = woe_list.loc[this_Fea]
    this_SplitNum = len(this_SplitPoint)-1
    
    this_ScoreCard = pd.DataFrame(np.zeros([this_SplitNum, 2]))
    
    if this_FeaType=='float64' or this_FeaType=='int64':
        for j in range(0, this_SplitNum):
            if j == this_SplitNum-1:
                this_ScoreCard.iloc[j,0]='('+str(this_SplitPoint[j])+','+str(this_SplitPoint[j+1])+')'
            else:
                this_ScoreCard.iloc[j,0]='('+str(this_SplitPoint[j])+','+str(this_SplitPoint[j+1])+']'
            this_ScoreCard.iloc[j,1] = -this_WOE[j]*this_Coef*B
        
    else:
        this_TranRule = x_train_category_tranRules[ x_train_category_tranRules_feaName.loc[this_Fea] ]
        
        for j in range(0, this_SplitNum):
            this_Bin_Raw = this_TranRule.iloc[ np.where((this_TranRule.loc[:, 'Transform data']>this_SplitPoint[j])&(this_TranRule.loc[:, 'Transform data']<=this_SplitPoint[j+1]))[0], 0 ]
            
            this_ScoreCard.iloc[j, 0] = this_Bin_Raw.iloc[0]
            for k in range(1, len(this_Bin_Raw)):
                this_ScoreCard.iloc[j,0] = this_ScoreCard.iloc[j,0] + ', ' + this_Bin_Raw.iloc[k]
            
            this_ScoreCard.iloc[j, 1] = -this_WOE[j]*this_Coef*B

    ScoreCard[i] = this_ScoreCard
    
#%%
#----------------------- 十二. 评估模型表现 -----------------------#

# 1. Evaluate ScoreCard Performance On Train Data
#section_num_train=[]
x_train_reindex = x_train.copy()
x_train_reindex.index = list(range(0, len(x_train_reindex)))
Data_X = x_train_reindex
Data_types = x_train_types
Data_Y = y_train_bin_reindex

score_samples = np.ones([len(Data_Y),])*score0 # Initialize Each Observation's Score

for i in range(0, chosen_feature_num):
    this_feature = chosen_feature_final[i]
    this_feature_type = Data_types.loc[this_feature]
    this_scorecard = ScoreCard[i]
    this_split_num = this_scorecard.shape[0]
    
    for j in range(0,this_scorecard.shape[0]):
        if this_feature_type=='float64' or this_feature_type=='int64': #数值型和类别型的加分方式不同
            dot_position = this_scorecard.iloc[j,0].find(',')
            lower_bound = float( this_scorecard.iloc[j,0][1:dot_position] ) #提取数值型分段的下界(从1开始以避开括号)
            higher_bound = float( this_scorecard.iloc[j,0][dot_position+1:-1] ) #提取数值型分段的上界   
            # np.where什么都没找到的话下面这句话等于无效，也可以用if来框住可读性会好一些
            score_samples[np.where( (Data_X.loc[:,this_feature]>lower_bound)&(Data_X.loc[:,this_feature]<=higher_bound) )] += this_scorecard.iloc[j,1]   
            #section_num_train.append(len(np.where((Data_X.loc[:,this_feature]>lower_bound-eps)&(Data_X.loc[:,this_feature]<higher_bound))[0]))
        else:
            this_section_types = this_scorecard.iloc[j,0]
            while this_section_types.find(',')>0: #若能找到分隔符，持续循环
                dot_position = this_section_types.find(',')
                this_type = this_section_types[:dot_position] #到当前逗号前的那个类
                score_samples[ np.where(Data_X.loc[:,this_feature]==this_type) ] += this_scorecard.iloc[j,1]
                this_section_types = this_section_types[dot_position+1:]#删掉已匹配完的类型可取值          
            #进行到这一步时，this_section_types只剩下一个可选值
            score_samples[np.where(Data_X.loc[:,this_feature]==this_section_types)] += this_scorecard.iloc[j,1]
            
score_samples = np.array(score_samples)
            
ks_value, bad_percent, good_percent = cal_ks(score_samples, Data_Y, section_num=20)

score_samples_train = score_samples

plt.figure()
plt.plot(list(range(0,21)),np.append([0],bad_percent),'-r',label='Bad Percent')
plt.plot(list(range(0,21)),np.append([0],good_percent),'-g',label='Good Percent')
plt.plot(list(range(0,21)),np.append([0],ks_value),'-b',label='KS value')
plt.legend(loc='lower right')
plt.ylabel('% of total Good/Bad')
plt.xlabel('% of population')

plt.figure()
predictions = 1-1/(np.exp((A-score_samples_train)/B)+1)
false_positive_rate, recall, thresholds = roc_curve(Data_Y, predictions)
roc_auc = auc(false_positive_rate, recall)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()            

print('评分卡在训练集打分的ks最大值为{0}'.format(np.max(ks_value)))

# 2. Evaluate ScoreCard Performance On Test Data
#section_num_train=[]
x_test_reindex = x_test.copy()
x_test_reindex.index = list(range(0, len(x_test_reindex)))
Data_X = x_test_reindex
Data_types = x_test_types
Data_Y = y_test_bin_reindex

score_samples = np.ones([len(Data_Y),])*score0 # Initialize Each Observation's Score

for i in range(0, chosen_feature_num):
    this_feature = chosen_feature_final[i]
    this_feature_type = Data_types.loc[this_feature]
    this_scorecard = ScoreCard[i]
    this_split_num = this_scorecard.shape[0]
    
    for j in range(0,this_scorecard.shape[0]):
        if this_feature_type=='float64' or this_feature_type=='int64': #数值型和类别型的加分方式不同
            dot_position = this_scorecard.iloc[j,0].find(',')
            lower_bound = float( this_scorecard.iloc[j,0][1:dot_position] ) #提取数值型分段的下界(从1开始以避开括号)
            higher_bound = float( this_scorecard.iloc[j,0][dot_position+1:-1] ) #提取数值型分段的上界   
            # np.where什么都没找到的话下面这句话等于无效，也可以用if来框住可读性会好一些
            score_samples[np.where( (Data_X.loc[:,this_feature]>lower_bound)&(Data_X.loc[:,this_feature]<=higher_bound) )] += this_scorecard.iloc[j,1]   
            #section_num_train.append(len(np.where((Data_X.loc[:,this_feature]>lower_bound-eps)&(Data_X.loc[:,this_feature]<higher_bound))[0]))
        else:
            this_section_types = this_scorecard.iloc[j,0]
            while this_section_types.find(',')>0: #若能找到分隔符，持续循环
                dot_position = this_section_types.find(',')
                this_type = this_section_types[:dot_position] #到当前逗号前的那个类
                score_samples[ np.where(Data_X.loc[:,this_feature]==this_type) ] += this_scorecard.iloc[j,1]
                this_section_types = this_section_types[dot_position+1:]#删掉已匹配完的类型可取值          
            #进行到这一步时，this_section_types只剩下一个可选值
            score_samples[np.where(Data_X.loc[:,this_feature]==this_section_types)] += this_scorecard.iloc[j,1]
            
score_samples = np.array(score_samples)
            
ks_value, bad_percent, good_percent = cal_ks(score_samples, Data_Y, section_num=20)

score_samples_test = score_samples

plt.figure()
plt.plot(list(range(0,21)),np.append([0],bad_percent),'-r',label='Bad Percent')
plt.plot(list(range(0,21)),np.append([0],good_percent),'-g',label='Good Percent')
plt.plot(list(range(0,21)),np.append([0],ks_value),'-b',label='KS value')
plt.legend(loc='lower right')
plt.ylabel('% of total Good/Bad')
plt.xlabel('% of population')

plt.figure()
predictions = 1-1/(np.exp((A-score_samples_test)/B)+1)
false_positive_rate, recall, thresholds = roc_curve(Data_Y, predictions)
roc_auc = auc(false_positive_rate, recall)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()            

print('评分卡在测试集打分的ks最大值为{0}'.format(np.max(ks_value)))

# 3. 计算模型PSI
PSI_value = PSI(score_samples_train, score_samples_test)

score_all = np.append(score_samples_train, score_samples_test)
Y_all = np.append(y_train_bin_reindex, y_test_bin_reindex)

ks_value, bad_percent, good_percent = cal_ks(score_all, Y_all)

plt.figure()

plt.hist(score_samples_train)
plt.ylabel('Number of samples')
plt.xlabel('Score')
plt.title('Distribution on train samples')

plt.figure()
plt.hist(score_samples_test)
plt.ylabel('Number of samples')
plt.xlabel('Score')
plt.title('Distribution on test samples')

print('评分卡在训练集和测试集上打分的PSI为{0}'.format(PSI_value))

#%%
#----------------------- 十三. 导出评分卡 -----------------------#
for i in range(0, len(chosen_feature_final)):
    this_feature = chosen_feature_final[i]
    this_scorecard = ScoreCard[i]
    section_num = this_scorecard.shape[0]
    this_feature_multi_en = pd.DataFrame( [this_feature for j in range(0, section_num)] )
    #chosen_final_ch_name = names.loc[this_feature]
        
    #this_feature_multi_ch = pd.DataFrame( [chosen_final_ch_name for j in range(0,section_num)], index=list(range(0,section_num)) )
    #this_output_scorecard = pd.concat([this_feature_multi_en,this_feature_multi_ch,this_scorecard],axis=1)
    this_output_scorecard = pd.concat([this_feature_multi_en, this_scorecard], axis=1)
    #this_output_scorecard.columns=['feature_en','feature_ch','section','score']
    this_output_scorecard.columns=['feature_en', 'section', 'score']
    
    if i==0:
        output_scorecard = this_output_scorecard
    else:
        output_scorecard = output_scorecard.append(this_output_scorecard)

'''输出路径，按照实际情况调整'''
out_path='C:/Users/Administrator/.spyder-py3/CIB_FINTECH_test/ScoreCard.csv'
output_scorecard.to_csv(out_path,encoding="utf_8_sig")






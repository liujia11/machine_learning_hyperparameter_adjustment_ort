# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:30:03 2018

@author: jia.liu
"""

import time
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns',20)

pd.set_option('display.width',60)

class AdjustParam(object):
    '''
    model：最优模型
    ort_res: 正交实验结果
    param_res：各因素水平结果
    '''
    def __init__(self, X, y, scoring=None, cv=5):
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv = cv
        self.model = None
        self.ort_res = None
        self.param_res = None
    
    def run_ort(self, model, df_params):
        model_str = str(model)
        k=0
        n=len(df_params.index)
        df_params['ort_res'] = 0
        df_params['ort_res_std'] = 0
        df_params['ort_train_time'] = 0
        for i in df_params.index:
            k+=1
            params_li = list(map(lambda x:x+'='+ repr(df_params.loc[i, x]),df_params.columns[:-3]))
            param_str = ', '.join(params_li)
            print(param_str)
            for re_p in params_li:
                p_val = re_p.split('=')[0]
                model_str = re.sub('%s=[^,)]*'%p_val, re_p ,model_str)
            model_ = eval(model_str)
            t1=time.time()
            cv_score = cross_val_score(model_, self.X, self.y, cv = 5, n_jobs = -1, scoring='neg_mean_squared_error' )
            err = np.sqrt(-cv_score)
            res = np.mean(err)
            res_std = np.std(cv_score)
            t2 = int(time.time()-t1)
            print('res: %f, time: %d, num: %d/%d'%(res,t2,k,n))
            df_params.loc[i,'ort_res'] = res
            df_params.loc[i,'ort_res_std'] = res_std
            df_params.loc[i,'ort_train_time'] = t2
        self.ort_res = df_params
        # 筛选最优值
        res_li = list(map(lambda x: df_params.groupby(x).ort_res.mean().idxmin(), df_params.columns[:-3]))
        param_li = df_params.columns[:-3]
        param_opt = list(map(lambda x: '='.join(x), zip(param_li,map(repr,res_li))))
        # 最优RF模型
        for re_p in param_opt:
            p_val = re_p.split('=')[0]
            model_str = re.sub('%s=[^,)]*'%p_val, re_p ,model_str)
        self.model = eval(model_str)

        temp_li = []
        for i in df_params.columns[:-3]:
            temp = df_params.groupby(i).mean()[['ort_res','ort_res_std','ort_train_time']].sort_values(['ort_res','ort_res_std','ort_train_time'])
            temp = temp.rename(index = lambda x: str(i) + '=' + repr(x))
            temp_li.append(temp)
        self.param_res = pd.concat(temp_li)

#%%
if __name__ == '__main__':
    from xgboost.sklearn import XGBRegressor
    from sklearn.model_selection import train_test_split
    from my_ort import ORT
    
    def rmsle_cv(model, n_folds=5):
        rmse= np.sqrt(-cross_val_score(model, X.values, y, scoring="neg_mean_squared_error", cv = n_folds))
        return rmse
    
    X = pd.read_csv('X.csv')
    y = pd.read_csv('y.csv')
    training_X,validation_X, training_y,validation_y = train_test_split(X, y ,test_size=0.2, random_state=17)
    xgb_model = XGBRegressor(n_jobs=-1)  
    # 构建正交实验表
    ort = ORT()
    # 输入参数数量，返回可选的正交表
    ort.seeSets(4,see_num=10)
    # 设计参数
    params = {'max_depth':[3],
    'gamma':[0.3, 0.4, 0.5],
    'colsample_bytree':[0.2],
    'n_estimators':[750],
    'subsample':[0.5, 0.7, 0.9],
    'reg_lambda':[1,5,10],
    'learning_rate':[0.01,0.1,0.3]}
    # 获取参数正交表
    param_df=ort.genSets(params,mode=2)
    # 训练模型
    a = AdjustParam(X, y, 'neg_mean_squared_error')
    a.run_ort(model=xgb_model, df_params=param_df)
    
    # 查看正交搜索自动选出的模型
    a.model
    # 查看所有的结果
    a.ort_res
    # 查看极差分析结果（可以根据正交实验的结果和训练时间，手动调整超参数）
    a.param_res
    

    # 正交分析选出的模型
    score = rmsle_cv(a.model)
    print("Averaged models error: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    # rmse: 0.1384 (0.0148)
    # 耗时 : 
    
    # 不进行正交分析，从测试的9组中选择表现最好的（相当于随机搜索）
    '''
       gamma  subsample  reg_lambda  learning_rate  max_depth  colsample_bytree  \
    0    0.3        0.5           1           0.01          3               0.2   
    1    0.3        0.7          10           0.10          3               0.2   
    2    0.3        0.9           5           0.30          3               0.2   
    3    0.4        0.5          10           0.30          3               0.2   
    4    0.4        0.7           5           0.01          3               0.2   
    5    0.4        0.9           1           0.10          3               0.2   
    6    0.5        0.5           5           0.10          3               0.2   
    7    0.5        0.7           1           0.30          3               0.2   
    8    0.5        0.9          10           0.01          3               0.2   
    
       n_estimators   ort_res  ort_res_std  ort_train_time  
    0           750  0.144111     0.004829              12  
    1           750  0.141269     0.004349              10  
    2           750  0.149688     0.004706              10  
    3           750  0.159380     0.005539              11  
    4           750  0.148800     0.005136              11  
    5           750  0.142483     0.004669              10  
    6           750  0.147529     0.004737              10  
    7           750  0.151033     0.004418              11  
    8           750  0.152610     0.005360              10
    '''
    # 第2组均方损失最小（序号1）
    # rmse: 0.141269
    # 通过正交分析选出的模型效果 比以上任意一组效果都好
    #%%
    # 随机搜索 & 网格搜索
    from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
    param_test = {
    'gamma':[0.3, 0.4, 0.5],
    'subsample':[0.5, 0.7, 0.9],
    'reg_lambda':[1,5,10],
    'learning_rate':[0.01,0.1,0.3]}
#    param_test = {'gamma':[0.3, 0.4, 0.5]}
    xgb_model = XGBRegressor(
        max_depth = 3,
        colsample_bytree = 0.2,
        learning_rate =0.1, 
        n_estimators = 750,
        n_jobs=-1
    )
    
    # 随机搜索
    t0 = time.time()
    rand = RandomizedSearchCV(xgb_model, param_test, cv=5, scoring='neg_mean_squared_error', n_iter=9, random_state=1)  #
    rand.fit(X, y)
    print('随机搜索-度量记录：',rand.cv_results_)  # 包含每次训练的相关信息
    print('随机搜索-最佳度量值:',np.sqrt(-rand.best_score_))  # 获取最佳度量值
    print('随机搜索-最佳参数：',rand.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
    print('随机搜索-最佳模型：',rand.best_estimator_)  # 获取最佳度量时的分类器模型
    t_rand = time.time() - t0
    # rmse: 0.14600703568644044
    # 随机搜索-最佳参数： {'subsample': 0.5, 'reg_lambda': 5, 'learning_rate': 0.1, 'gamma': 0.4}
    # 耗时 70
    
    # 网格搜索
    gsearch = GridSearchCV( xgb_model , param_grid = param_test, scoring='neg_mean_squared_error', cv=5 )
    t1 = time.time()
    gsearch.fit(X, y)
    gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
    t_gsearch = time.time()-t1
    print(t_gsearch)
    print('随机搜索-度量记录：',gsearch.cv_results_)  # 包含每次训练的相关信息
    print('随机搜索-最佳度量值:',np.sqrt(-gsearch.best_score_))  # 获取最佳度量值
    print('随机搜索-最佳参数：',gsearch.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
    print('随机搜索-最佳模型：',gsearch.best_estimator_)  # 获取最佳度量时的分类器模型
    # rmse ：0.13812885469181446
    # 网格搜索 最佳参数：{'gamma': 0.3, 'learning_rate': 0.1, 'reg_lambda': 1, 'subsample': 0.5}
    # 耗时 ：636s
    '''
    总结
               rmse   用时（秒）
    正交分析  0.1384   95
    网格搜索  0.1381   636
    随机搜索  0.1460   70
    
    网格搜索搜索了所有情况的参数组合, 损失肯定是最小的，但却需要进行3^4 共81组训练
    随机搜索的结果和随机数的选取有关
    正交分析能在较短时间内得到趋于最优值的结果，效果显著，训练组数和随机搜索一样，结果趋近于网格搜索
    （耗时略比随机搜索高的原因在于代码可能未经过优化，或选取的参数不同训练时长不一样）
    '''

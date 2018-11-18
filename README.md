@[TOC](机器学习调参 一种超参数调整方法的尝试（附测试链接）)

# 简介
下面的内容是本人想法的一次尝试性的工作，本文的目的仅在于学习交流，有的知识也是一知半解，不对的地方尽请指正。
通过设计正交实验，对结果进行极差分析，选择超参数的值，迭代这个过程，逐渐找到更精细的值。以下简称正交搜索。

目前较常用的超参数调整方法是网格搜索和随机搜索。
网格搜索：
优点：能得到更准确的结果。
缺点：当待调整的超参数较多时，需要进行大量组合。

随机搜索：
优点：计算量小。
缺点：最终的结果和搜索的次数以及随机种子的选取有关，搜索次数不够时可能得不到较好的结果。

正交搜索：
优点：计算量小，可以得到相对准确的结果。同时优化多个超参数。
缺点：需要进行超参数取值进行设计。
解决：相关的设计，代码中可以进行查询，根据正交表，设置超参数范围。

# 调参流程
以xgboost的超参数调整为例（数据见后面的项目链接）
```python
from xgboost.sklearn import XGBRegressor
xgb_model = XGBRegressor(n_jobs=-1)  
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')
```
## 1. 设计超参数
查看正交表（比如有四个超参数）
```python
# my_ort代码后面的链接中会有
from my_ort import ORT
# 构建正交实验表
ort = ORT()
# 输入参数数量，返回可选的正交表
ort.seeSets(4,see_num=10)
```
```
Out[3]: 
                0  num
1     2^4 4^1 n=8    5
2         3^4 n=9    4
3       2^11 n=12   11
4    2^4 3^1 n=12    5
6    2^8 8^1 n=16    9
7        4^5 n=16    5
8    3^6 6^1 n=18    7
9       2^19 n=20   19
10   2^8 5^1 n=20    9
12  2^20 4^1 n=24   21
解释：
例：
2^4 4^1 n=8    5
2^4 4^1 表示有四个超参数的取值有两个，1个超参数的取值有4个
n=8 表示需要进行8组实验（8次搜索）
num=5 表示最多可以有5个超参数
```
有4个超参数，可以根据正交表中的3^4设置参数，
即这4个超参数有3个取值
```python
params = {
'gamma':[0.3, 0.4, 0.5],
'subsample':[0.5, 0.7, 0.9],
'reg_lambda':[1,5,10],
'learning_rate':[0.01,0.1,0.3],
'max_depth':[3],
'colsample_bytree':[0.2],
'n_estimators':[750]
}
# 获取参数正交表
param_df=ort.genSets(params,mode=2)
```
```
Out[12]: 
   gamma  subsample  reg_lambda  learning_rate  max_depth  colsample_bytree  n_estimators
0    0.3        0.5           1           0.01          3               0.2           750
1    0.3        0.7          10           0.10          3               0.2           750
2    0.3        0.9           5           0.30          3               0.2           750
3    0.4        0.5          10           0.30          3               0.2           750
4    0.4        0.7           5           0.01          3               0.2           750
5    0.4        0.9           1           0.10          3               0.2           750
6    0.5        0.5           5           0.10          3               0.2           750
7    0.5        0.7           1           0.30          3               0.2           750
8    0.5        0.9          10           0.01          3               0.2           750
```
## 3. 进行训练
```python
# 训练模型 传入训练集，对应的标签，以及损失函数 AdjustParam类见链接代码
a = AdjustParam(X, y, 'neg_mean_squared_error')
# 开始训练 传入模型，设计好的超参数正交表
a.run_ort(model=xgb_model, df_params=param_df)
```
```python
# 查看正交搜索自动选出的模型
a.model
```
```
Out[15]: 
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.2, gamma=0.3, learning_rate=0.1,
       max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
       n_estimators=750, n_jobs=-1, nthread=None, objective='reg:linear',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=0.7)
```
```python
# 查看所有的结果
a.ort_res
```
```
   gamma  subsample  reg_lambda  learning_rate  max_depth  colsample_bytree  n_estimators   ort_res  ort_res_std  ort_train_time
0    0.3        0.5           1           0.01          3               0.2           750  0.144111     0.004829               9
1    0.3        0.7          10           0.10          3               0.2           750  0.141269     0.004349              10
2    0.3        0.9           5           0.30          3               0.2           750  0.149688     0.004706              10
3    0.4        0.5          10           0.30          3               0.2           750  0.159380     0.005539               9
4    0.4        0.7           5           0.01          3               0.2           750  0.148800     0.005136               9
5    0.4        0.9           1           0.10          3               0.2           750  0.142483     0.004669               8
6    0.5        0.5           5           0.10          3               0.2           750  0.147529     0.004737               9
7    0.5        0.7           1           0.30          3               0.2           750  0.151033     0.004418               9
8    0.5        0.9          10           0.01          3               0.2           750  0.152610     0.005360               8
ort_res  ：均方根误差
ort_res_std ：交叉验证的方差
ort_train_time ：交叉验证消耗的训练时间
```
```python
# 查看极差分析结果（可以根据正交实验的结果和训练时间，手动调整超参数）
a.param_res
```
```
Out[27]: 
                       ort_res  ort_res_std  ort_train_time
gamma=0.3             0.145023     0.004628        9.666667
gamma=0.4             0.150221     0.005115        8.666667
gamma=0.5             0.150390     0.004838        8.666667
subsample=0.7         0.147034     0.004634        9.333333
subsample=0.9         0.148260     0.004912        8.666667
subsample=0.5         0.150340     0.005035        9.000000
reg_lambda=1          0.145876     0.004639        8.666667
reg_lambda=5          0.148672     0.004860        9.333333
reg_lambda=10         0.151086     0.005083        9.000000
learning_rate=0.1     0.143760     0.004585        9.000000
learning_rate=0.01    0.148507     0.005109        8.666667
learning_rate=0.3     0.153367     0.004888        9.333333
max_depth=3           0.148545     0.004860        9.000000
colsample_bytree=0.2  0.148545     0.004860        9.000000
n_estimators=750      0.148545     0.004860        9.000000
每个超参数对应的最小的ort_res，就是最后确定的超参数值
```
注：当使用得分优化模型时（分值越大越好）需要对AdjustParam中的代码进行修改
## 4. 结果验证
```python
def rmsle_cv(model, n_folds=5):
    rmse= np.sqrt(-cross_val_score(model, X.values, y, scoring="neg_mean_squared_error", cv = n_folds))
    return rmse
# 正交分析选出的模型
score = rmsle_cv(a.model)
print("error: {:.4f} ({:.4f})".format(score.mean(), score.std()))
```
```
error: 0.1384 (0.0148)
得到的结果优于前面训练的9组数据
```

## 5. 随机搜索&网格搜索
```python
# 随机搜索 & 网格搜索
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
param_test = {
'gamma':[0.3, 0.4, 0.5],
'subsample':[0.5, 0.7, 0.9],
'reg_lambda':[1,5,10],
'learning_rate':[0.01,0.1,0.3]}
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
```
## 6. 正交搜索VS随机搜索VS网格搜索
```
           rmse   用时（秒）
正交分析  0.1384   95
网格搜索  0.1381   636
随机搜索  0.1460   70
```
## 7. 结论
   网格搜索：搜索了所有情况的参数组合, 损失最小，但却需要进行3^4 共81组训练
    随机搜索：结果和随机数的选取有关
    正交搜索：能在较短时间内得到趋于最优值的结果，效果显著，训练组数和随机搜索一样，结果趋近于网格搜索。
## 8. 更多分析
还可对正交表做更多的挖掘，比如在损失相当的情况下，可选择训练速度更快的模型。
# 传送门
## 本文测试样例链接（含代码及数据）
发布到CSDN上的链接：https://blog.csdn.net/LittleDonkey_Python/article/details/84203295
## 其它
正交表构造参考：https://github.com/lovesoo/OrthogonalArrayTest 
添加功能（函数seeSets）：输入变量数，返回可选的正交表结构

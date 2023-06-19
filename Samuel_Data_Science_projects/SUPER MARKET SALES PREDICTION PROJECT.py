import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import reduce_memory_usage
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from xgboost import plot_importance
from prettytable import PrettyTable
import pickle
pd.set_option('display.float_format',lambda x:f'{x:.3f}')
#pd.set_option('display.max_columns', None)

# 1.
dftrain=pd.read_csv('D:\\Downloads\\bigmart\\sales_train.csv')
dftest = pd.read_csv('D:\\Downloads\\bigmart\\test.csv')
dfitems = pd.read_csv('D:\\Downloads\\bigmart\\items.csv')
print(dftrain.shape)
print(dftrain.head())
print(dftrain.info()) #to see datatype of each column
print(dftrain.describe())
print(dftrain.isnull().sum()/len(dftrain) *100)
print('number of duplicates:',len(dftrain[dftrain.duplicated()]))

# 2.date column into date datatype and ordering as date
dftrain['date'] = pd.to_datetime(dftrain['date'],format='%d.%m.%Y')
dftrain.sort_values(by='date',inplace=True)
print(dftrain.head())
print(dftrain.date.min(),'to',dftrain.date.max())

dftrain2 = dftrain.merge(dfitems,on='item_id',how ='left')
dftest2 = dftest.merge(dfitems,on='item_id',how='left')
print(len(dftrain.shop_id.unique()))
print(len(dftrain.item_id.unique()))
print(len(dftrain2.item_category_id.unique()))
print(dftrain2.head())

# 3.outliers in item_cnt_day
print(dftrain.item_cnt_day.describe())
outlier_item_cnt = np.percentile(dftrain.item_cnt_day.values,100)
print(outlier_item_cnt )
for i in range(90,101):
    print('{}th percentile value of item count:{}'.format(i,np.percentile(dftrain.item_cnt_day,i)))
# so it is clear that 99% of data has 5 or less item_cnt_day
# only 1 % has more than 5 item_cnt_day

# outliers in item price column:
for i in range(90,101):
    print('{}th percentile value of item price:{}'.format(i,np.percentile(dftrain.item_price,i)))

# to see shopwise selling count
dftrain3= dftrain.groupby(['shop_id'],as_index=False)['item_cnt_day'].sum()
print(len(dftrain3))
plt.figure(figsize=(20,4))
sns.barplot(dftrain3,x='shop_id',y='item_cnt_day')
plt.show()
# to see monthwise selling
dftrain3= dftrain.groupby(['date_block_num'],as_index=False)['item_cnt_day'].sum()
print(len(dftrain3))
plt.figure(figsize=(20,4))
sns.barplot(dftrain3,x='date_block_num',y='item_cnt_day')
plt.show()


# 4.remove shops,items not in test dataset b/c training only wanted details gives high accuracy
test_shop = dftest['shop_id'].unique()
test_item = dftest['item_id'].unique()
print(len(test_shop),'**',len(test_item))
_train = dftrain[dftrain['shop_id'].isin(test_shop)]
_train = _train[_train['item_id'].isin(test_item)]
print(_train.shape)
print(_train.describe())

# 5.outlier removal:
train =_train[(_train.item_price<5000) & (_train.item_cnt_day<1000)]
# removing negative values
train= train[(train.item_price>0) & (train.item_cnt_day>=0)].reset_index(drop=True)
print(train.shape)

# Cleaning shops data
# We have different shop ids for same shop names like 0-57, 1-58, 11,10, 40-39
# We dont have 0,1,11 and 40 shop_id in test data so we are replacing these with shop_id which shares a similar names
train.loc[train.shop_id == 0, 'shop_id'] = 57
train.loc[train.shop_id == 1, 'shop_id'] = 58
train.loc[train.shop_id == 11, 'shop_id'] = 10
train.loc[train.shop_id == 40, 'shop_id'] = 39

train = train.merge(dfitems,on='item_id',how ='left')
print(train.head())

# Aranging columns
#single line code to change order of columns in one df and save it in new df:
train_monthly = train[['date', 'date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'item_cnt_day']]
train_monthly['revenue'] = train_monthly['item_price'] * train_monthly['item_cnt_day']

#6. Group by month in this case "date_block_num" and aggregate features.
# here groupby multiple columns used so it creates hierarchy
train_monthly = train_monthly.sort_values('date').groupby(['date_block_num',
                                                           'shop_id', 'item_category_id', 'item_id'], as_index=False)
#print(train_monthly.head())
train_monthly = train_monthly.agg({ 'item_cnt_day':['sum', 'mean'],
                                    'item_price':['mean'], 'revenue':['sum']})
#print(train_monthly)

# Rename columns
train_monthly.columns = ['date_block_num', 'shop_id', 'item_category_id',
                         'item_id', 'item_cnt_month', 'mean_item_cnt', 'mean_item_price', 'revenue_month']

#7.building a new data set with all the possible combinations of [‘date_block_num’, ‘shop_id’, ‘item_id’] so we won’t have missing records.
shop_ids = dftest['shop_id'].unique()
item_ids = dftest['item_id'].unique()
all_posible = []
for i in tqdm(range(34)):        #tqdm is used to show progress speed of this loop
    for shop in shop_ids:
        for item in item_ids:
            all_posible.append([i, shop, item])
empty_df = pd.DataFrame(all_posible, columns=['date_block_num','shop_id','item_id'])
print('no of rows before concatination:',len(empty_df))
print(empty_df.head(30))

# Create a test set for month 34.
dftest["date_block_num"] = 34
dftest["date_block_num"] = 34
dftest["date_block_num"] = dftest["date_block_num"].astype(np.int8)
dftest["shop_id"] = dftest.shop_id.astype(np.int8)
dftest["item_id"] = dftest.item_id.astype(np.int16)

# adding 34th month data(dftest) into allpossible dataset(emptydf):
# rowwise concating empty and test dataframe:
empty_df = pd.concat([empty_df, dftest.drop(["ID"],axis = 1)],  
                     ignore_index=True, sort=False, 
                     keys=["date_block_num", "shop_id", "item_id"])
print(empty_df.isnull().sum()) # it shows there is no null value in each column.
empty_df.fillna( 0, inplace = True )
print('no of rows after concatination:',len(empty_df))

#8.Here I am extracting various mean encoding features of shop, item , item category, date block .

# Merge the train set with the complete set and fill missing records with 0
train_monthly = pd.merge(empty_df, train_monthly, on=['date_block_num','shop_id','item_id'], how='left')
train_monthly.fillna(0, inplace=True)
print('length of trainmonthly:',len(train_monthly))
print(train_monthly.columns)
print(train_monthly.isnull().sum())

# since item category id becomes null values, add it again for test block 34
train_monthly = train_monthly.drop(['item_category_id'], axis=1).join(dfitems, on='item_id', rsuffix='_').drop(['item_id_','item_name'], axis=1)
# Aranging columns
train_monthly = train_monthly[['date_block_num', 'shop_id', 'item_category_id',
                               'item_id', 'revenue_month', 'item_cnt_month', 'mean_item_cnt','mean_item_price']]
#print(train_monthly.isnull().sum()/len(train_monthly)*100)
print(train_monthly.isnull().sum())

train_monthly['item_cnt_month'] = train_monthly['item_cnt_month'].fillna(0).clip(0, 20)

# 9.Extract time based features (month)
# in lambda if num cannot div by 12, then it returns original num.
# eg: 0%12=0,2%12=2,12%12=0,13%12=1,33%12=9
train_monthly['month'] = train_monthly['date_block_num'].apply(lambda x: (x % 12))

# Add date_block_shop_mean:
date_block_shop_mean = train_monthly.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
#print(date_block_shop_mean)
date_block_shop_mean.columns = ['date_block_shop_mean'] # it changes last column name
#print(date_block_shop_mean)
date_block_shop_mean.reset_index(inplace=True) # it creates new index no
#print(date_block_shop_mean)
train_monthly = pd.merge(train_monthly, date_block_shop_mean, on=['date_block_num', 'shop_id'], how='left')

# ADD date_block_item_mean:
date_block_item_mean = train_monthly.groupby(['date_block_num','item_id']).agg({'item_cnt_month': ['mean']})
date_block_item_mean.columns = ['date_block_item_mean']
date_block_item_mean.reset_index(inplace=True)
train_monthly = pd.merge(train_monthly, date_block_item_mean, on=['date_block_num','item_id'], how='left')

# Add date_category_mean
date_category_mean = train_monthly.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
date_category_mean.columns = ['date_cat_mean']
date_category_mean.reset_index(inplace=True)
train_monthly = pd.merge(train_monthly, date_category_mean, on=['date_block_num', 'item_category_id'], how='left')

train_monthly.fillna(0, inplace=True)
train_monthly = reduce_memory_usage.reduce_memory_usage(train_monthly)
print(train_monthly.head())

# Define function to compute lag features
def lag_feature( df,lags, cols ):
    for col in cols:
        tmp = df[["date_block_num", "shop_id","item_id",col ]]
        print(col)
        for i in tqdm(range(1,lags+1)):
            shifted = tmp.copy()
            shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag_"+str(i)] # it renames columns
            shifted.date_block_num = shifted.date_block_num + i  # it adds [1,2,3] to all values in date_block_num
            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

# Add lag features
train_monthly = lag_feature( train_monthly, 3 , ['item_cnt_month','revenue_month',
                                                 'mean_item_cnt','mean_item_price',
                                                 'date_block_shop_mean','date_block_item_mean',
                                                 'date_cat_mean'] )

# fill null values with zeros and optimize the memory
train_monthly = train_monthly.fillna(0)
train_monthly = reduce_memory_usage.reduce_memory_usage(train_monthly)
#pd.set_option('display.max_columns', None)
print(train_monthly.columns)
print(train_monthly.shape)

# find mean,std,min,max for 3 lag columns we created
# columnwise mean calculation(3 lag columns)
# Add quarter mean count
train_monthly['qmean_item_cnt'] = train_monthly[['item_cnt_month_lag_1',
                                    'item_cnt_month_lag_2',
                                    'item_cnt_month_lag_3']].mean(skipna=True, axis=1)
# Add quarter std count
train_monthly['qstd_item_cnt'] = train_monthly[['item_cnt_month_lag_1',
                                    'item_cnt_month_lag_2',
                                    'item_cnt_month_lag_3']].std(skipna=True, axis=1)
# Add quarter min count
train_monthly['qmin_item_cnt'] = train_monthly[['item_cnt_month_lag_1',
                                    'item_cnt_month_lag_2',
                                    'item_cnt_month_lag_3']].min(skipna=True, axis=1)
# Add quarter max count
train_monthly['qmax_item_cnt'] = train_monthly[['item_cnt_month_lag_1',
                                    'item_cnt_month_lag_2',
                                    'item_cnt_month_lag_3']].max(skipna=True, axis=1)

print(train_monthly.shape)

# create input and target
X = train_monthly.drop(['item_cnt_month','mean_item_price','revenue_month','mean_item_cnt', 'date_block_shop_mean','date_block_item_mean','date_cat_mean'], axis=1)
y = train_monthly['item_cnt_month']

#Normalize Dataset
scaler = MinMaxScaler()
for col in tqdm(X.columns[4:]):
    X[col] = scaler.fit_transform(X[col].values.reshape(-1,1))

# saving final preprosessed data into drive
X = reduce_memory_usage.reduce_memory_usage(X)
with open('D:\\preprocessed_set2','wb') as loc:
    pickle.dump((X,y),loc)


# Train test split:
# Load preprocessed data
with open('D:\\preprocessed_set2','rb') as loc:
    X, y = pickle.load(loc)

# slpitting data into train,validation and test
# omit first 3 months b/c they dont hv lag values.
X_train = X[X['date_block_num']<33]
X_train = X_train[X['date_block_num']>2] #boolean result from x used in xtrain dataframe.
y_train = y[X['date_block_num']<33]
y_train = y_train[X['date_block_num']>2]

X_val = X[X['date_block_num']==33]
y_val = y[X['date_block_num']==33]

X_test = X[X['date_block_num']==34]
print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape)

#LASSO REGRESSION:
'''
alpha = [10**i for i in range(-6,2)]
train_score =[]
val_score=[]
for i in tqdm(alpha):
    print('alpha={}'.format(i))
    _model = Lasso(alpha=i)
    _model.fit(X_train,y_train)
    rmse_train = mean_squared_error(_model.predict(X_train).clip(0,20),y_train,squared=False)
    rmse_val = mean_squared_error(_model.predict(X_val).clip(0,20),y_val,squared=False)

    if val_score:
        if sorted(val_score)[0]>rmse_val:
            print('model saving....')
            with open('D:\\best_lasso','wb') as loc:
                pickle.dump(_model,loc)
    else:
        print('model saving...')
        with open('D:\\best_lasso','wb') as loc:
            pickle.dump(_model,loc)

    train_score.append(rmse_train)
    val_score.append((rmse_val))
    print('Training loss is {}'.format(rmse_train))
    print('Validation loss is {}'.format(rmse_val))
    print('-'*50)
with open('D:\\lasso_log','wb') as loc:
    pickle.dump((train_score,val_score,alpha),loc)
'''
print("-"*50)
with open('D:\\best_lasso','rb') as loc:
    best_lasso = pickle.load(loc)
with open('D:\\lasso_log','rb') as loc:
    train_score,val_score,alpha=pickle.load(loc)

print('train_score for lasso:',train_score)
print('val_score for lasso:',val_score)
print('alpha for lasso:',alpha)
'''
params = [str(i) for i in alpha]
fig,ax = plt.subplots(figsize=(13,6)) # figsize(width,height)
ax.plot(params,val_score,c='r') # line graph
for i, j in enumerate(np.round(val_score,3)): #(ennumerate gives no. to each iteraton and bind them)
    ax.annotate((params[i],j),(params[i],val_score[i])) #annotate gives notes to points on graph
plt.grid()
plt.title('Cross validation rmse vs alpha for Lasso')
plt.xlabel('alpha values')
plt.ylabel('rmse')
plt.show()
'''
# Ridge Regression
'''
alpha = [10 ** i for i in range(-4, 2)]
train_score = []
val_score = []
for i in tqdm(alpha):

    print("alpha = {} ".format(i))
    _model = Ridge(alpha=i)
    _model.fit(X_train, y_train)
    rmse_train = mean_squared_error(_model.predict(X_train).clip(0, 20), y_train, squared=False)
    rmse_val = mean_squared_error(_model.predict(X_val).clip(0, 20), y_val, squared=False)

    if val_score:
        if sorted(val_score)[0] > rmse_val:
            print("model saving.....")
            with open('D:\\best_ridge', 'wb') as loc:
                pickle.dump(_model, loc)
    else:
        print("model saving.....")
        with open('D:\\best_ridge', 'wb') as loc:
            pickle.dump(_model, loc)

    train_score.append(rmse_train)
    val_score.append(rmse_val)
    print("Training Loss is {} ".format(rmse_train))
    print("Validation Loss is {} ".format(rmse_val))
    print("-" * 50)

with open('D:\\ridge_log', 'wb') as loc:
    pickle.dump((train_score, val_score, alpha), loc)
'''
print("-"*50)
with open('D:\\ridge_log','rb') as loc:
    train_score,val_score,alpha=pickle.load(loc)
print('train_score for Ridge:',train_score)
print('val_score for Ridge:',val_score)
print('alpha for Ridge:',alpha)

'''
with open('D:\\best_ridge','rb') as loc:
    best_ridge = pickle.load(loc)
with open('D:\\ridge_log','rb') as loc:
    train_score,val_score,alpha = pickle.load(loc)
params = [str(i) for i in alpha]
fig, ax = plt.subplots(figsize=(15,6))
ax.plot(params, val_score,c='g')
for i, txt in enumerate(np.round(val_score,3)):
    ax.annotate((params[i],np.round(txt,3)), (params[i],val_score[i]))
plt.grid()
plt.title("Cross Validation rmse for para grid")
plt.xlabel("(subsample , cosample_bytree)")
plt.ylabel("Error measure")
plt.show()
'''

'''
# DecisionTreeRegressor
alpha = [3, 5, 7, 9, 12]
train_score = []
val_score = []
for i in tqdm(alpha):

    print("alpha = {} ".format(i))
    _model = DecisionTreeRegressor(max_depth=i)
    _model.fit(X_train, y_train)
    rmse_train = mean_squared_error(_model.predict(X_train).clip(0, 20), y_train, squared=False)
    rmse_val = mean_squared_error(_model.predict(X_val).clip(0, 20), y_val, squared=False)

    if val_score:
        if sorted(val_score)[0] > rmse_val:
            print("model saving.....")
            with open('D:\\best_dt', 'wb') as loc:
                pickle.dump(_model, loc)
    else:
        print("model saving.....")
        with open('D:\\best_dt', 'wb') as loc:
            pickle.dump(_model, loc)

    train_score.append(rmse_train)
    val_score.append(rmse_val)
    print("Training Loss is {} ".format(rmse_train))
    print("Validation Loss is {} ".format(rmse_val))
    print("-" * 50)

with open('D:\\dt_log', 'wb') as loc:
    pickle.dump((train_score, val_score, alpha), loc)
'''
print("-"*50)
with open('D:\\dt_log','rb') as loc:
    train_score,val_score,alpha=pickle.load(loc)
print('train_score for DecisionTree:',train_score)
print('val_score for DecisionTree:',val_score)
print('alpha for DecisionTree:',alpha)
'''
# line graph rmse vs alpha
with open('D:\\best_dt','rb') as loc:
    best_dt = pickle.load(loc)
with open('D:\\dt_log','rb') as loc:
    train_score,val_score,alpha = pickle.load(loc)
params = [str(i) for i in alpha]
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(params, val_score,c='g')
for i, txt in enumerate(np.round(val_score,3)):
    ax.annotate((params[i],np.round(txt,3)), (params[i],val_score[i]))
plt.grid()
plt.title("Cross Validation rmse for para grid")
plt.xlabel("(subsample , cosample_bytree)")
plt.ylabel("Error measure")
plt.show() 
# Graph for feature importance in DecisionTree
print(best_dt.feature_importances_)
feat_importances = pd.DataFrame(best_dt.feature_importances_, index=X.columns, columns=["Importance"])
print(feat_importances)
feat_importances.sort_values(by='Importance', ascending=True, inplace=True)
feat_importances.plot.barh(figsize=(14,6))
plt.show()
'''

'''
# RandomForestRegressor
alpha = [3, 5, 7, 9]
train_score = []
val_score = []
for i in tqdm(alpha):

    print("alpha = {} ".format(i))
    _model = RandomForestRegressor(max_depth=i)
    _model.fit(X_train, y_train)
    rmse_train = mean_squared_error(_model.predict(X_train).clip(0, 20), y_train, squared=False)
    rmse_val = mean_squared_error(_model.predict(X_val).clip(0, 20), y_val, squared=False)

    if val_score:
        if sorted(val_score)[0] > rmse_val:
            print("model saving.....")
            with open('D:\\best_rf', 'wb') as loc:
                pickle.dump(_model, loc)
    else:
        print("model saving.....")
        with open('D:\\best_rf', 'wb') as loc:
            pickle.dump(_model, loc)

    train_score.append(rmse_train)
    val_score.append(rmse_val)
    print("Training Loss is {} ".format(rmse_train))
    print("Validation Loss is {} ".format(rmse_val))
    print("-" * 50)

with open('D:\\rf_log', 'wb') as loc:
    pickle.dump((train_score, val_score, alpha), loc)
'''
print("-"*50)
with open('D:\\best_rf','rb') as loc:
    best_rf = pickle.load(loc)
with open('D:\\rf_log','rb') as loc:
    train_score,val_score,alpha=pickle.load(loc)
print('train_score for Randomforest:',train_score)
print('val_score for RandomForest:',val_score)
print('alpha for Randomforest:',alpha)
'''
# line graph for rmse vs alpha
params = [str(i) for i in alpha]
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(params, val_score,c='g')
for i, txt in enumerate(np.round(val_score,3)):
    ax.annotate((params[i],np.round(txt,3)), (params[i],val_score[i]))
plt.grid()
plt.title("Cross Validation rmse for para grid")
plt.xlabel("(subsample , cosample_bytree)")
plt.ylabel("Error measure")
plt.show()
# graph for feature importance:
feat_importances = pd.DataFrame(best_rf.feature_importances_, index=X.columns, columns=["Importance"])
feat_importances.sort_values(by='Importance', ascending=True, inplace=True)
feat_importances.plot.barh(figsize=(8,6))
plt.show()
'''

# XGBoost Regressor_1:
# Custom Grid search Part 1 with max_depth and min_child_weight:
'''
max_depth = [5, 9, 10, 14]
min_child_weight = [1, 6,10]

fit_params={"early_stopping_rounds":20,
            "eval_metric" : "rmse",
            "eval_set" : [[X_val, y_val]] }

train_score = []
val_score = []
params = []
for i in max_depth:
    for j in min_child_weight:
        print("Tuning parameter max_depth and min_child_weight")
        print("set max_depth = {} and min_child_weight = {}".format(i,j))
        _model = XGBRegressor(eta = 0.1, n_estimators=1000,max_depth=i ,min_child_weight=j, verbose=100)
        _model.fit(X_train, y_train, verbose=True, **fit_params)
        rmse_train = mean_squared_error(_model.predict(X_train).clip(0,20),y_train, squared=False)
        rmse_val = mean_squared_error(_model.predict(X_val).clip(0,20),y_val, squared=False)

        if val_score:
            if sorted(val_score)[0] > rmse_val:
                print("model saving.....")
                with open('D:\\best_xgb_1','wb') as loc:
                    pickle.dump(_model,loc)
        else:
            print("model saving.....")
            with open('D:\\best_xgb_1','wb') as loc:
                pickle.dump(_model,loc)

        train_score.append(rmse_train)
        val_score.append(rmse_val)
        params.append((i,j))

        print("Training Loss when max_depth={} and min_child_weight={} is {} ".format(i,j,rmse_train))
        print("Validation Loss when max_depth={} and min_child_weight={} is {} ".format(i,j,rmse_val))
        print("-"*50)

with open('D:\\xgb_log_1','wb') as loc:
    pickle.dump((train_score,val_score,params),loc)
'''
print("-"*50)
with open('D:\\best_xgb_1','rb') as loc:
    best_xgb_1 = pickle.load(loc)
with open('D:\\xgb_log_1','rb') as loc:
    train_score,val_score,params=pickle.load(loc)
print('train_score for xgb1:',train_score)
print('val_score for xgb1:',val_score)
print('params for xgb1:',params)
'''
# line graph for rmse vs params
params = [str(i) for i in params]
fig, ax = plt.subplots(figsize=(13,8))
ax.plot(params, val_score,c='g')
for i, txt in enumerate(np.round(val_score,3)):
    ax.annotate((params[i],np.round(txt,3)), (params[i],val_score[i]))
plt.grid()
plt.title("Cross Validation rmse for para grid")
plt.xlabel("(max_depth , min_child_weight)")
plt.ylabel("Error measure")
plt.show()

# Plot feature importance
def plot_features(booster, figsize):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    plot_importance(booster=booster, ax=ax, grid=False)
    return plt.show()

plot_features(best_xgb_1, (10,10))
# We get the best score with max_depth = 10 and min_child_weight = 6.
'''
# XGBoost Regressor_2:
# Custom Grid search Part 2 with subsample and cosample_bytree:
'''
subsample = [1, 0.8, 0.6, 0.3]
cosample_bytree = [1, 0.8, 0.6, 0.3]
fit_params={"early_stopping_rounds":10,
            "eval_metric" : "rmse",
            "eval_set" : [[X_val, y_val]] }

best_param_1 = params[np.argmin(val_score)] # it gives index of min value

train_score_2 = []
val_score_2 = []
params_2 = []
for i in subsample:
    for j in cosample_bytree:
        print("Tuning parameter subsamle and cosample_bytree")
        print("set max_depth = {} and min_child_weight = {}".format(best_param_1[0],best_param_1[1]))
        print("set subsamle = {} and cosample_bytree = {}".format(i,j))
        _model = XGBRegressor(eta = 0.1, n_estimators=1000,
                              max_depth=best_param_1[0] ,
                              min_child_weight=best_param_1[1],
                              subsample=i,colsample_bytree=j)
        _model.fit(X_train, y_train, verbose=True, **fit_params)
        rmse_train = mean_squared_error(_model.predict(X_train).clip(0,20),y_train, squared=False)
        rmse_val = mean_squared_error(_model.predict(X_val).clip(0,20),y_val, squared=False)

        if val_score_2:
            if sorted(val_score_2)[0] > rmse_val:
                print("saving model.....")
                with open('D:\\best_xgb_2','wb') as loc:
                    pickle.dump(_model,loc)
        else:
            print("saving model.....")
            with open('D:\\best_xgb_2','wb') as loc:
                pickle.dump(_model,loc)

        train_score_2.append(rmse_train)
        val_score_2.append(rmse_val)
        params_2.append((i,j))

        print("Training Loss when subsample={} and cosample_bytree={} is {} ".format(i,j,rmse_train))
        print("Validation Loss when subsample={} and cosample_bytree={} is {} ".format(i,j,rmse_val))
        print("-"*50)
with open('D:\\xgb_log_2','wb') as loc:
    pickle.dump((train_score_2,val_score_2,params_2),loc)
'''
print("-"*50)
with open('D:\\best_xgb_2','rb') as loc:
    best_xgb_2 = pickle.load(loc)
with open('D:\\xgb_log_2','rb') as loc:
    train_score_2,val_score_2,params_2=pickle.load(loc)
print('train_score for xgb2:',train_score_2)
print('val_score for xgb2:',val_score_2)
print('params for xgb2:',params_2)
'''
# line graph for rmse vs params
params = [str(i) for i in params_2]
fig, ax = plt.subplots(figsize=(13,10))
ax.plot(params, val_score_2,c='r')
for i, txt in enumerate(np.round(val_score_2,3)):
    ax.annotate((params[i],np.round(txt,3)), (params[i],val_score_2[i]))
plt.grid()
plt.title("Cross Validation rmse for para grid")
plt.xlabel("(subsample , cosample_bytree)")
plt.ylabel("Error measure")
plt.show()
# feature importance:
def plot_features(booster, figsize):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    plot_importance(booster=booster, ax=ax, grid=False)
    return plt.show()

plot_features(best_xgb_2, (10,12))
'''

'''
# Training
xgb_final = XGBRegressor(
    max_depth=5,
    n_estimators=1000,
    min_child_weight=1,
    colsample_bytree=0.3,
    subsample=1,
    eta=0.1,
    seed=42)

xgb_final.fit(
    X_train,
    y_train,
    eval_metric="rmse",
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True,
    early_stopping_rounds = 20)

rmse_train_final = mean_squared_error(xgb_final.predict(X_train).clip(0,20),y_train, squared=False)
rmse_val_final = mean_squared_error(xgb_final.predict(X_val).clip(0,20),y_val, squared=False)



with open('D:\\xgb_final','wb') as loc:
    pickle.dump(xgb_final,loc)
with open('D:\\xgb_final_log_','wb') as loc:
    pickle.dump((rmse_train_final,rmse_val_final),loc)
'''
print("-"*50)
with open('D:\\xgb_final_log_','rb') as loc:
    rmse_train_final,rmse_val_final=pickle.load(loc)
print('rmse_train_xgbfinal:',rmse_train_final)
print('rmse_val_xgbfinal:',rmse_val_final)


with open('D:\\best_lasso','rb') as loc:
    best_lasso = pickle.load(loc)
with open('D:\\lasso_log','rb') as loc:
    lasso_log = pickle.load(loc)
with open('D:\\best_ridge','rb') as loc:
    best_ridge = pickle.load(loc)
with open('D:\\ridge_log','rb') as loc:
    ridge_log = pickle.load(loc)
with open('D:\\best_dt','rb') as loc:
    best_dt = pickle.load(loc)
with open('D:\\dt_log','rb') as loc:
    dt_log = pickle.load(loc)
with open('D:\\best_rf','rb') as loc:
    best_rf = pickle.load(loc)
with open('D:\\rf_log','rb') as loc:
    rf_log = pickle.load(loc)
with open('D:\\best_xgb_1','rb') as loc:
    best_xgb_1 = pickle.load(loc)
with open('D:\\xgb_log_1','rb') as loc:
    xgb_log_1 = pickle.load(loc)
with open('D:\\best_xgb_2','rb') as loc:
    best_xgb_2 = pickle.load(loc)
with open('D:\\xgb_log_2','rb') as loc:
    xgb_log_2 = pickle.load(loc)
with open('D:\\xgb_final_log_','rb') as loc:
    xgb_final_log=pickle.load(loc)


x = PrettyTable()
x.field_names = ["Model", "train_rmse" , "val_rmse"]
x.add_row(["Lasso Regression", round(lasso_log[0][np.argmin(lasso_log[1])],5), round(min(lasso_log[1]),5)])
x.add_row(["Ridge Regression", round(ridge_log[0][np.argmin(ridge_log[1])],5), round(min(ridge_log[1]),5)])
x.add_row(["Decision tree Regressor", round(dt_log[0][np.argmin(dt_log[1])],5), round(min(dt_log[1]),5)])
x.add_row(["Random Forest Regressor", round(rf_log[0][np.argmin(rf_log[1])],5), round(min(rf_log[1]),5)])
x.add_row(["XGB Regressor 1", round(xgb_log_1[0][np.argmin(xgb_log_1[1])],5), round(min(xgb_log_1[1]),5)])
x.add_row(["XGB Regressor 2", round(xgb_log_2[0][np.argmin(xgb_log_2[1])],5), round(min(xgb_log_2[1]),5)])
x.add_row(['XGB REGRESSOR FINAL',round(xgb_final_log[0],5),round(xgb_final_log[1],5)])
print(x)

# predicting any value
def predict_sales(shop_id, item_id):
    with open('D:\\xgb_final', 'rb') as loc:
        xgb_final = pickle.load(loc)
    try:
        pred = xgb_final.predict(X_test[X_test.shop_id == shop_id][X_test.item_id == item_id]).clip(0, 20)
        return print('prediction : {}'.format(int(pred)))
    except:
        return print("Please enter valid shop_id/item_id")

predict_sales(5,5002)


def predict_score(shop_id, item_id):
    with open('D:\\xgb_final', 'rb') as loc:
        best_model = pickle.load(loc)
    try:
        pred = best_model.predict(X_val[X_val.shop_id == shop_id][X_val.item_id == item_id]).clip(0, 20)
        y = y_val[X_val.shop_id == shop_id][X_val.item_id == item_id]
        rmse = mean_squared_error(pred, y, squared=False)
        mse = mean_squared_error(pred, y, squared=True)
        print('RMSE score : {}'.format(mean_squared_error(pred, y, squared=False)))
        print('RMSE:',rmse)
        print('MSE:',mse)
    except:
        return print("Please enter valid shop_id/item_id")
predict_score(5,5002)

'''with open('D:\\preprocessed_set2','rb') as loc:
    X, y = pickle.load(loc)
print(y.unique())
print(y.value_counts())'''


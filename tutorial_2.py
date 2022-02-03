import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
%matplotlib inline

items = pd.read_csv("original_datas/items.csv")
item_category = pd.read_csv("original_datas/item_categories.csv")
shops = pd.read_csv("original_datas/shops.csv")
train = pd.read_csv("original_datas/sales_train.csv")
test = pd.read_csv("original_datas/test.csv")

item_category['big_category_name'] = item_category['item_category_name'].map(lambda x: x.split(' - ')[0])

item_category.loc[item_category['big_category_name']=='Чистые носители (штучные)',
    'big_categry'
] = 'Чистые носители (шпиль)'

shops["city_name"] = shops["shop_name"].map(lambda x: x.split(" ")[0])

shops.loc[shops['city_name']=='!Якутск','city_name'] = 'Якутск'

train['date_sales'] = train['item_cnt_day'] * train['item_price']

mon_shop_item_cnt = train[
    ['date_block_num','shop_id','item_id','item_cnt_day']
].groupby(
    ['date_block_num','shop_id','item_id'],
    as_index=False
).sum().rename(columns={'item_cnt_day':'mon_shop_item_cnt'})

mon_shop_item_sales = train[
    ['date_block_num','shop_id','item_id','date_sales']
].groupby(
    ['date_block_num','shop_id','item_id'],
    as_index=False
).sum().rename(columns={'date_sales':'mon_shop_item_sales'})

train_full_comb = pd.DataFrame()
for i in range(35):
    mid = test[['shop_id','item_id']]
    mid['date_block_num'] = i
    train_full_comb = pd.concat([train_full_comb,mid],axis=0)
    
train_ = pd.merge(
    train_full_comb,
    mon_shop_item_cnt,
    on=['date_block_num','shop_id','item_id'],
    how='left'
)

train_ = pd.merge(
    train_,
    mon_shop_item_sales,
    on=['date_block_num','shop_id','item_id'],
    how='left'
)

train_ = pd.merge(
    train_,
    items[['item_id','item_category_id']],
    on='item_id',
    how='left'
)

train_ = pd.merge(
    train_,
    item_category[['item_category_id','big_category_name']],
    on='item_category_id',
    how='left'
)

train_ = pd.merge(
    train_,
    shops[['shop_id','city_name']],
    on='shop_id',
    how='left'
)

plt_df = train_.groupby(
    ['date_block_num'],
    as_index=False
).sum()
plt.figure(figsize=(20, 10))
sns.lineplot(x='date_block_num',y='mon_shop_item_cnt',data=plt_df)
plt.title('Montly item counts')

plt_df = train_.groupby(
    ['date_block_num','big_category_name'],
    as_index=False
).sum()
plt.figure(figsize=(20, 10))
sns.lineplot(x='date_block_num',y='mon_shop_item_cnt',data=plt_df,hue='big_category_name')
plt.title('Montly item counts by big category')

plt_df = train_.groupby(
    ['date_block_num','city_name'],
    as_index=False
).sum()
plt.figure(figsize=(20, 10))
sns.lineplot(x='date_block_num',y='mon_shop_item_cnt',data=plt_df,hue='city_name')
plt.title('Montly item counts by city_name')

train_['mon_shop_item_cnt'] = train_['mon_shop_item_cnt'].clip(0,20)

lag_col_list = ['mon_shop_item_cnt','mon_shop_item_sales']
lag_num_list = [1,3,6,9,12]

train_ = train_.sort_values(
    ['shop_id', 'item_id','date_block_num'],
    ascending=[True, True,True]
).reset_index(drop=True)

for lag_col in lag_col_list:
    for lag in lag_num_list:
        set_col_name =  lag_col + '_' +  str(lag)
        df_lag = train_[['shop_id', 'item_id','date_block_num',lag_col]].sort_values(
            ['shop_id', 'item_id','date_block_num'],
            ascending=[True, True,True]
        ).reset_index(drop=True).shift(lag).rename(columns={lag_col: set_col_name})
        train_ = pd.concat([train_, df_lag[set_col_name]], axis=1)
        
train_ = train_.fillna(0)

train_ = train_[(train_['date_block_num']<=33) & (train_['date_block_num']>=12)].reset_index(drop=True)
test_ = train_[train_['date_block_num']==34].reset_index(drop=True)

train_y = train_['mon_shop_item_cnt']
train_X = train_.drop(columns=['date_block_num','mon_shop_item_cnt', 'mon_shop_item_sales'])
test_X = test_.drop(columns=['date_block_num','mon_shop_item_cnt', 'mon_shop_item_sales'])

from sklearn.preprocessing import LabelEncoder
obj_col_list = ['big_category_name','city_name']
for obj_col in obj_col_list:
    le = LabelEncoder()
    train_X[obj_col] = pd.DataFrame({obj_col:le.fit_transform(train_X[obj_col])})
    test_X[obj_col] = pd.DataFrame({obj_col:le.fit_transform(test_X[obj_col])})
    
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(train_X,train_y)

plt.figure(figsize=(20, 10))
sns.barplot(
    x=rfr.feature_importances_,
    y=train_X.columns.values
)
plt.title('Importance of features')

test_y = rfr.predict(test_X)
test_X['item_cnt_month'] = test_y
submission = pd.merge(
    test_,
    test_X[['shop_id','item_id','item_cnt_month']],
    on=['shop_id','item_id'],
    how='left'
)

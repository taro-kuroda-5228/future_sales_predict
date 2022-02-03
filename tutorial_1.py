import numpy as np
import pandas as pd
import datetime

items = pd.read_csv("original_datas/items.csv")
item_category = pd.read_csv("original_datas/item_categories.csv")
shops = pd.read_csv("original_datas/shops.csv")
train = pd.read_csv("original_datas/sales_train.csv")
test = pd.read_csv("original_datas/test.csv")

wk = pd.merge(
    train,
    items,
    on='item_id',
    how='left'
)

wk = pd.merge(
    wk,
    item_category,
    on='item_category_id',
    how='left'
)

wk = pd.merge(
    wk,
    shops,
    on='shop_id',
    how='left'
)

test = pd.merge(
    test,
    items,
    on='item_id',
    how='left'
)

test = pd.merge(
    test,
    item_category,
    on='item_category_id',
    how='left'
)

test = pd.merge(
    test,
    shops,
    on='shop_id',
    how='left'
)

wk.loc[:,'date'] = wk.loc[:,'date'].map(
    lambda x:
    datetime.date(
        datetime.datetime.strptime(x, '%d.%m.%Y').year,
        datetime.datetime.strptime(x, '%d.%m.%Y').month,
        datetime.datetime.strptime(x, '%d.%m.%Y').day
    )
)

wk['year'] = wk.loc[:,'date'].map(lambda x: x.year)
wk['month'] = wk.loc[:,'date'].map(lambda x: x.month)
wk['day'] = wk.loc[:,'date'].map(lambda x: x.day)

grouped = wk[wk['month']==12].groupby(['shop_id','item_id'], as_index=False)
test_pred_nov_exist = grouped.mean()[['shop_id','item_id','item_cnt_day']]

pred = pd.merge(
    test,
    test_pred_nov_exist,
    on=['shop_id','item_id'],
    how='left'
)

pred = pred[['ID','shop_id','item_id','item_cnt_day']]
pred.loc[pred['item_cnt_day'].isnull(),'item_cnt_day'] = 0

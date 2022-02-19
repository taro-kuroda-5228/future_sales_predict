import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from movwin import MovingWindowKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "original_datas"


def read_csv(file_name: str) -> pd.DataFrame:
    """Read csv files by putting file names."""
    file = pd.read_csv(f"{DATA_PATH}/{file_name}.csv")
    return file


file_names = ["items", "item_categories", "shops", "sales_train", "test"]
for file_name in file_names:
    locals()[file_name] = read_csv(file_name)

item_categories["big_category_name"] = [
    x.split(" - ")[0] for x in item_categories["item_category_name"]
]


shops["city_name"] = [x.split(" ")[0] for x in shops["shop_name"]]
shops.loc[shops["city_name"] == "!Якутск", "city_name"] = "Якутск"

sales_train["date_sales"] = sales_train["item_cnt_day"] * sales_train["item_price"]


def mon_shop_item(date_data: str, mon_data: str) -> pd.DataFrame:
    """Get monthly sales data from dayly sales data."""
    df = (
        sales_train[["date_block_num", "shop_id", "item_id", date_data]]
        .groupby(["date_block_num", "shop_id", "item_id"], as_index=False)
        .sum()
        .rename(columns={date_data: mon_data})
    )
    return df


mon_shop_item_cnt = mon_shop_item("item_cnt_day", "mon_shop_item_cnt")
mon_shop_item_sales = mon_shop_item("date_sales", "mon_shop_item_sales")

train_full_comb = pd.DataFrame()
for i in range(35):
    mid = test[["shop_id", "item_id"]]
    mid["date_block_num"] = i
    train_full_comb = pd.concat([train_full_comb, mid], axis=0)


def merge(
    merging_df_1: pd.DataFrame, merging_df_2: pd.DataFrame, merged_column_lists: list
) -> pd.DataFrame:
    """Merge 2 pd.DataFrame on a columns list of merged_column_lists."""
    df = pd.merge(merging_df_1, merging_df_2, on=merged_column_lists, how="left")
    return df


train = merge(
    train_full_comb, mon_shop_item_cnt, ["date_block_num", "shop_id", "item_id"]
)
train = merge(train, mon_shop_item_sales, ["date_block_num", "shop_id", "item_id"])
train = merge(train, items[["item_id", "item_category_id"]], "item_id")
train = merge(
    train,
    item_categories[["item_category_id", "big_category_name"]],
    "item_category_id",
)
train = merge(train, shops[["shop_id", "city_name"]], "shop_id")
train = train.fillna(0)
train["mon_shop_item_cnt"] = train["mon_shop_item_cnt"].clip(0, 20)

lag_col_list = ["mon_shop_item_cnt", "mon_shop_item_sales"]
lag_num_list = [1, 3, 6, 9, 12]

for lag_col in lag_col_list:
    for lag in lag_num_list:
        set_col_name = lag_col + "_" + str(lag)
        df_lag = (
            train[["shop_id", "item_id", "date_block_num", lag_col]]
            .sort_values(
                ["shop_id", "item_id", "date_block_num"], ascending=[True, True, True]
            )
            .reset_index(drop=True)
            .shift(lag)
            .rename(columns={lag_col: set_col_name})
        )
        train = pd.concat([train, df_lag[set_col_name]], axis=1)

train_ = train[train["date_block_num"] <= 33].reset_index(drop=True)
test_ = train[train["date_block_num"] == 34].reset_index(drop=True)

obj_col_list = ["big_category_name", "city_name"]
for obj_col in obj_col_list:
    le = LabelEncoder()
    train_[obj_col] = pd.DataFrame({obj_col: le.fit_transform(train_[obj_col])})
    test_[obj_col] = pd.DataFrame({obj_col: le.fit_transform(test_[obj_col])})

y = train_["mon_shop_item_cnt"]
X = train_.drop(columns=["mon_shop_item_cnt"]).fillna(0)

folds = MovingWindowKFold(ts_column="date_block_num", n_splits=5)
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

clf = lgb.LGBMRegressor()
scores_lgb = []
for train_index, test_index in folds.split(X):
    X_train, X_test = (
        X.iloc[
            train_index,
        ],
        X.iloc[
            test_index,
        ],
    )
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    scores_lgb.append([mae, mse])


def mean_scores(scores_list):
    """Get mean MAE and MSE scores in 5 times tests.
    Args:
        score_list: results of 5 time tests, shape of [mae,mse]
    """
    sum = 0
    for i in range(len(scores_list)):
        sum += scores_list[i][0]
    mae_avg = sum / 5

    sum = 0
    for i in range(len(scores_list)):
        sum += scores_list[i][1]
    mse_avg = sum / 5
    avg_list = [mae_avg, mse_avg]
    return avg_list


mean_score = mean_scores(scores_lgb)
print(f"MAE:{mean_score[0]}, MSE:{mean_score[1]}")
# MAE:0.05164586260217431, MSE:0.12601170078959073

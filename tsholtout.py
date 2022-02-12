from calendar import month_name

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split

register_matplotlib_converters()


def ts_train_test_split(df, ts_column, **options):
    """時系列情報が含まれるカラムでソートした iloc を返す Hold-Out"""
    # シャッフルしない
    options["shuffle"] = False
    # 時系列のカラムを取り出す
    ts = df[ts_column]
    # 元々のインデックスを振り直して iloc として使える値 (0, 1, 2...) にする
    ts_df = ts.reset_index()
    # 時系列でソートする
    sorted_ts_df = ts_df.sort_values(by=ts_column)
    # 添字を計算する
    train_index, test_index = train_test_split(sorted_ts_df.index, **options)
    return list(train_index), list(test_index)


def main():
    df = sns.load_dataset("flights")

    month_name_3_digits = [month_name[i][:3] for i in range(13)]
    month_name_mappings = {
        name: str(n).zfill(2) for n, name in enumerate(month_name_3_digits)
    }
    df["month"] = df["month"].apply(lambda x: month_name_mappings[x])
    df["year-month"] = df.year.astype(str) + "-" + df.month.astype(str)
    df["year-month"] = pd.to_datetime(df["year-month"], format="%Y-%m")

    # データの並び順をシャッフルする
    df = df.sample(frac=1.0, random_state=42)

    # 学習データとテストデータに分割する
    train_index, test_index = ts_train_test_split(
        df, ts_column="year-month", test_size=0.33
    )

    # 添字
    print(f"index of train: {train_index}")
    print(f"index of test: {test_index}")

    # グラフに描いてみる
    sns.lineplot(
        data=df.iloc[train_index], x="year-month", y="passengers", label="train"
    )
    sns.lineplot(data=df.iloc[test_index], x="year-month", y="passengers", label="test")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

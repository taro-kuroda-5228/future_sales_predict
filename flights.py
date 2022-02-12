from calendar import month_name

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def main():
    # 航空機の旅客数を記録したデータセットを読み込む
    df = sns.load_dataset("flights")

    # 時系列のカラムを用意する
    month_name_3_digits = [month_name[i][:3] for i in range(13)]
    month_name_mappings = {
        name: str(n).zfill(2) for n, name in enumerate(month_name_3_digits)
    }
    df["month"] = df["month"].apply(lambda x: month_name_mappings[x])
    df["year-month"] = df.year.astype(str) + "-" + df.month.astype(str)
    df["year-month"] = pd.to_datetime(df["year-month"], format="%Y-%m")

    # データの並び順を元に分割する
    folds = TimeSeriesSplit(n_splits=5)

    # 5 枚のグラフを用意する
    fig, axes = plt.subplots(5, 1, figsize=(12, 12))

    # 学習用のデータとテスト用のデータに分割するためのインデックス情報を得る
    for i, (train_index, test_index) in enumerate(folds.split(df)):
        # 生のインデックス
        print(f"index of train: {train_index}")
        print(f"index of test: {test_index}")
        print("----------")
        # 元のデータを描く
        sns.lineplot(
            data=df, x="year-month", y="passengers", ax=axes[i], label="original"
        )
        # 学習用データを描く
        sns.lineplot(
            data=df.iloc[train_index],
            x="year-month",
            y="passengers",
            ax=axes[i],
            label="train",
        )
        # テスト用データを描く
        sns.lineplot(
            data=df.iloc[test_index],
            x="year-month",
            y="passengers",
            ax=axes[i],
            label="test",
        )

    # グラフを表示する
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

import sys
from calendar import month_name

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

class MovingWindowKFold(TimeSeriesSplit):
    """時系列情報が含まれるカラムでソートした iloc を返す KFold"""

    def __init__(self, ts_column, clipping=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 時系列データのカラムの名前
        self.ts_column = ts_column
        # 得られる添字のリストの長さを過去最小の Fold に揃えるフラグ
        self.clipping = clipping

    def split(self, X, *args, **kwargs):
        # 渡されるデータは DataFrame を仮定する
        assert isinstance(X, pd.DataFrame)

        # clipping が有効なときの長さの初期値
        train_fold_min_len, test_fold_min_len = sys.maxsize, sys.maxsize

        # 時系列のカラムを取り出す
        ts = X[self.ts_column]
        # 元々のインデックスを振り直して iloc として使える値 (0, 1, 2...) にする
        ts_df = ts.reset_index()
        # 時系列でソートする
        sorted_ts_df = ts_df.sort_values(by=self.ts_column)
        # スーパークラスのメソッドで添字を計算する
        for train_index, test_index in super().split(sorted_ts_df, *args, **kwargs):
            # 添字を元々の DataFrame の iloc として使える値に変換する
            train_iloc_index = sorted_ts_df.iloc[train_index].index
            test_iloc_index = sorted_ts_df.iloc[test_index].index

            if self.clipping:
                # TimeSeriesSplit.split() で返される Fold の大きさが徐々に大きくなることを仮定している
                train_fold_min_len = min(train_fold_min_len, len(train_iloc_index))
                test_fold_min_len = min(test_fold_min_len, len(test_iloc_index))

            yield list(train_iloc_index[-train_fold_min_len:]), list(test_iloc_index[-test_fold_min_len:])


def main():
    df = sns.load_dataset('flights')

    month_name_3_digits = [month_name[i][:3] for i in range(13)]
    month_name_mappings = {name: str(n).zfill(2) for n, name in enumerate(month_name_3_digits)}
    df['month'] = df['month'].apply(lambda x: month_name_mappings[x])
    df['year-month'] = df.year.astype(str) + '-' + df.month.astype(str)
    df['year-month'] = pd.to_datetime(df['year-month'], format='%Y-%m')

    # データの並び順をシャッフルする
    df = df.sample(frac=1.0, random_state=42)

    # 特定のカラムを時系列としてソートした分割
    folds = MovingWindowKFold(ts_column='year-month', n_splits=5)

    fig, axes = plt.subplots(5, 1, figsize=(12, 12))

    # 元々のデータを時系列ソートした iloc が添字として得られる
    for i, (train_index, test_index) in enumerate(folds.split(df)):
        print(f'index of train: {train_index}')
        print(f'index of test: {test_index}')
        print('----------')
        sns.lineplot(data=df, x='year-month', y='passengers', ax=axes[i], label='original')
        sns.lineplot(data=df.iloc[train_index], x='year-month', y='passengers', ax=axes[i], label='train')
        sns.lineplot(data=df.iloc[test_index], x='year-month', y='passengers', ax=axes[i], label='test')

    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
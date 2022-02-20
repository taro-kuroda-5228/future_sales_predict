folds = MovingWindowKFold(ts_column="date_block_num", n_splits=5)
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

clf = lgb.LGBMRegressor()
lr = LogisticRegression(max_iter=8000000)
scores_lgb = []
scores_lr = []
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

    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    scores_lr.append([mae, mse])

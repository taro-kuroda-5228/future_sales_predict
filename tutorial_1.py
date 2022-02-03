{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "55665e9a-95d6-407f-943d-2dd03d31ca29",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "34a9b622-9f35-4fdd-b8b3-65a32bc70597",
   "metadata": {},
   "outputs": [],
   "source": [
    "items = pd.read_csv(\"original_datas/items.csv\")\n",
    "item_category = pd.read_csv(\"original_datas/item_categories.csv\")\n",
    "shops = pd.read_csv(\"original_datas/shops.csv\")\n",
    "train = pd.read_csv(\"original_datas/sales_train.csv\")\n",
    "test = pd.read_csv(\"original_datas/test.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "d7135b47-58e8-4143-8397-009fbc24e0e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "wk = pd.merge(\n",
    "    train,\n",
    "    items,\n",
    "    on='item_id',\n",
    "    how='left'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "aa0d4d60-6cc2-40cf-9a5d-f2093abf6c34",
   "metadata": {},
   "outputs": [],
   "source": [
    "wk = pd.merge(\n",
    "    wk,\n",
    "    item_category,\n",
    "    on='item_category_id',\n",
    "    how='left'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f9600937-b651-455f-9f64-4fe6aa52ce81",
   "metadata": {},
   "outputs": [],
   "source": [
    "wk = pd.merge(\n",
    "    wk,\n",
    "    shops,\n",
    "    on='shop_id',\n",
    "    how='left'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8636683b-60b6-4c63-a7e6-484ffdcc48ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "test = pd.merge(\n",
    "    test,\n",
    "    items,\n",
    "    on='item_id',\n",
    "    how='left'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f7ef58a0-c0be-4c63-9333-d7356ace3b58",
   "metadata": {},
   "outputs": [],
   "source": [
    "test = pd.merge(\n",
    "    test,\n",
    "    item_category,\n",
    "    on='item_category_id',\n",
    "    how='left'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ac220496-e3d9-489e-82f7-26f7812a2306",
   "metadata": {},
   "outputs": [],
   "source": [
    "test = pd.merge(\n",
    "    test,\n",
    "    shops,\n",
    "    on='shop_id',\n",
    "    how='left'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "bd0e22ef-16f5-4721-b41e-822b0a5b0a22",
   "metadata": {},
   "outputs": [],
   "source": [
    "wk.loc[:,'date'] = wk.loc[:,'date'].map(\n",
    "    lambda x:\n",
    "    datetime.date(\n",
    "        datetime.datetime.strptime(x, '%d.%m.%Y').year,\n",
    "        datetime.datetime.strptime(x, '%d.%m.%Y').month,\n",
    "        datetime.datetime.strptime(x, '%d.%m.%Y').day\n",
    "    )\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "187876cd-13f2-43b7-bd72-062afda4a456",
   "metadata": {},
   "outputs": [],
   "source": [
    "wk['year'] = wk.loc[:,'date'].map(lambda x: x.year)\n",
    "wk['month'] = wk.loc[:,'date'].map(lambda x: x.month)\n",
    "wk['day'] = wk.loc[:,'date'].map(lambda x: x.day)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "c99bf090-0cdf-434f-b422-aad39709eabc",
   "metadata": {},
   "outputs": [],
   "source": [
    "grouped = wk[wk['month']==12].groupby(['shop_id','item_id'], as_index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "a17828b9-4626-4b4f-b14c-01e5dbcfac3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_pred_nov_exist = grouped.mean()[['shop_id','item_id','item_cnt_day']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "53526139-94dd-4bb4-84ef-df3c7856d4eb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>shop_id</th>\n",
       "      <th>item_id</th>\n",
       "      <th>item_cnt_day</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "      <td>32</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>33</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>482</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>485</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2</td>\n",
       "      <td>791</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111466</th>\n",
       "      <td>59</td>\n",
       "      <td>22087</td>\n",
       "      <td>1.928571</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111467</th>\n",
       "      <td>59</td>\n",
       "      <td>22088</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111468</th>\n",
       "      <td>59</td>\n",
       "      <td>22091</td>\n",
       "      <td>1.833333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111469</th>\n",
       "      <td>59</td>\n",
       "      <td>22092</td>\n",
       "      <td>1.666667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111470</th>\n",
       "      <td>59</td>\n",
       "      <td>22167</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>111471 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        shop_id  item_id  item_cnt_day\n",
       "0             2       32      1.000000\n",
       "1             2       33      1.000000\n",
       "2             2      482      1.000000\n",
       "3             2      485      1.000000\n",
       "4             2      791      1.000000\n",
       "...         ...      ...           ...\n",
       "111466       59    22087      1.928571\n",
       "111467       59    22088      2.000000\n",
       "111468       59    22091      1.833333\n",
       "111469       59    22092      1.666667\n",
       "111470       59    22167      1.000000\n",
       "\n",
       "[111471 rows x 3 columns]"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_pred_nov_exist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "4b513d9f-c34c-4e83-98dd-98733101ff43",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred = pd.merge(\n",
    "    test,\n",
    "    test_pred_nov_exist,\n",
    "    on=['shop_id','item_id'],\n",
    "    how='left'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "b4381549-3940-477e-bf83-82c68165254f",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred = pred[['ID','shop_id','item_id','item_cnt_day']]\n",
    "pred.loc[pred['item_cnt_day'].isnull(),'item_cnt_day'] = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d46d16c6-51fb-4662-bc9e-26037a55601e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

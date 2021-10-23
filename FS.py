# -*- coding = utf-8 -*-
# @Time :2021/10/15 12:50 下午
# @Author: XZL
# @File : FS.py
# @Software: PyCharm
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

# 删除sum为0和方差过小的数据
def delete_zero_and_var(df, var_throd):
    """

    :param df:  输入是训练集X
    :param var_throd:
    :return:
    """
    sum_zero_count = 0
    var_count = 0
    delete = []  # 删除要加上"SMILES"
    for index, row in df.iteritems():
        if index != "SMILES" and index != "pIC50":
            # 计算该列和
            row = (row - row.min()) / (row.max() - row.min())

            if df[index].sum() == 0:
                df[index] = row
                delete.append((index))
                sum_zero_count = sum_zero_count + 1
                continue

            if row.var() < var_throd:
                var_count = var_count + 1
                delete.append((index))

    final_sum = df.shape[1] - 1 - var_count - sum_zero_count
    print("最终剩余特征数", final_sum)
    print("sum=0特征: ", sum_zero_count, "方差过小特征: ", var_count)
    print(delete)
    # 删除sum为0后的数据
    handle_df = df.drop(delete, axis=1)
    return handle_df


# 随机森林特征提取
def feature_select(df, X, y, function_name, feature_num):
    """
    :param df:  所有总的DataFrame
    :param X:  需要筛选的X根据
    :param y: 需要X对应的y
    :param feature_num: 筛选的特征数量
    :return: 筛选后的DataFrame
    """
    feature_names = list(X)  # 所有列名

    if function_name == "RFR":
        rf = RandomForestRegressor(min_samples_split=6, n_estimators=100)
        rf.fit(X, y)
        # model的feature_importances_ 参数排序
        rank_impotant = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names), reverse=True)
        # 展示前20个
        RFR_ret = pd.DataFrame(rank_impotant).head(feature_num)
        return (df[list(RFR_ret[1])], list(RFR_ret[1]))

    if function_name == "Lasso":
        lasso = Lasso(alpha=.1)  # alpha越大，模型越稀疏，越多特征系数变为0，
        lasso.fit(X, y)
        ret = pretty_print_linear(lasso.coef_, feature_names, sort=True)
        lasso_ret = pd.DataFrame(ret).head(feature_num)
        return df[list(lasso_ret[1])], list(lasso_ret[1])

    if function_name == "RFE":
        lr = LinearRegression()
        # estimator估计函数
        rfe = RFE(estimator=lr, n_features_to_select=feature_num)
        rfe.fit(X, y)
        rfe_ret = pd.DataFrame(X.columns, index=rfe.ranking_, columns=['Rank']).sort_index(ascending=True).head(
            feature_num)
        return df[list(rfe_ret['Rank'])], list(rfe_ret['Rank'])


def pretty_print_linear(coefs, names=None, sort=False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: x[0], reverse=True)
    return lst

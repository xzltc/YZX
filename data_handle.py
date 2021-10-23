# -*- coding = utf-8 -*-
# @Time :2021/10/14 12:29 下午
# @Author: XZL
# @File : data_handle.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# xlxs转csv
def xlsx_to_csv_pd(xlsx_path, csv_path, sheet_name=0):
    data_xls = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    data_xls.to_csv(csv_path, encoding='utf-8')


if __name__ == '__main__':
    xlsx_path = './data/xlsx/Molecular_Descriptor.xlsx'
    csv_path = './data/csv/Molecular_Descriptor_test_Q3.csv'
    xlsx_to_csv_pd(xlsx_path, csv_path, 1)

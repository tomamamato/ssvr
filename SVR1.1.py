import streamlit as st
import pandas as pd
import numpy as np
from numpy.ma.testutils import assert_array_almost_equal
from sklearn.svm import SVR
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
import heapq
from collections import Counter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression
from skopt import BayesSearchCV
from sklearn.cross_decomposition import PLSRegression
from scipy.optimize import minimize
from sklearn.naive_bayes import GaussianNB
file_out='d3.xlsx'
file_out_in='d4.xlsx'
kk=[]
st.write("kk")
@st.cache_data
def xd0():
    excel_file = pd.ExcelFile(uploaded_file, engine="openpyxl")
    xls = pd.ExcelFile(excel_file)
    df = pd.read_excel(xls)
    columns = df.columns.tolist()
    n1 = len(columns)+1
    # 创建ExcelWriter对象以保存结果
    with pd.ExcelWriter(file_out) as writer:
        # 遍历每个sheet
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0, index_col=None)
            change_columns = {
                '发酵时间': columns[0]
            }
            for new_col, base_col in change_columns.items():
                df[new_col] = df[base_col].diff()
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return n1

@st.cache_data
def xd1():
    xls = pd.ExcelFile(file_out)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)

        # 获取第二列到第十二列的索引 (索引从0开始，因此第二列是索引1，第十二列是索引11)
        columns_to_process = df.columns[1:n1 - 1]

        # 计算差值并添加到DataFrame中
        for col in columns_to_process:
            # 计算差值，去掉第一行，因为没有上一行可以计算差值
            df[f'{col}_变化'] = df[col].diff()

        # 将处理后的DataFrame存储在字典中
        processed_sheets[sheet_name] = df

    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    st.success("已拓展“变化”特征")
@st.cache_data
def xd2():
    xls = pd.ExcelFile(file_out)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    n3 = len(columns)

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)
        for col in range(n1, n3):  # 13对应的是第14列，因为索引从0开始
            df[f'变化率_{df.columns[col]}'] = df.iloc[:, col] / df['发酵时间']  # 12对应的是第13列

        processed_sheets[sheet_name] = df

    # 将处理后的DataFrame写入新的Excel文件 'data3.xlsx'
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    st.success("已拓展“变化率”特征")

@st.cache_data
def xd3():
    xls = pd.ExcelFile(file_out)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    n4 = len(columns)

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)
        for col in range(1, n4):  # 第2列的索引是1，第35列的索引是34
            # 新列的值为上一行的数据
            df[f'上一_{df.columns[col]}'] = df.iloc[:, col].shift(1)

        processed_sheets[sheet_name] = df

    # 将处理后的DataFrame写入新的Excel文件 'data3.xlsx'
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    st.success("已拓展“时序”特征")
@st.cache_data
def xd4():
    xls = pd.ExcelFile(file_out)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    n5 = len(columns)

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)

        for col in range(1, n5):  # 索引从1开始，第二列是索引1，直到第69列索引68
            new_col_name = f"{df.columns[col]}_squared"  # 新特征列名称
            df[new_col_name] = df.iloc[:, col] ** 2  # 计算平方并添加新列

        processed_sheets[sheet_name] = df

    # 将处理后的DataFrame写入新的Excel文件 'data3.xlsx'
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    st.success("已拓展“多项式”特征")
@st.cache_data
def xd5():
    xls = pd.ExcelFile(file_out)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    n6 = len(columns)

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)

        if f'{df.columns[1]}_变化' in df.columns:
            df['耗糖累积量'] = df[f'{df.columns[1]}_变化'].cumsum()
        if '碱重kg_变化' in df.columns:
            df['耗碱累积量'] = df['碱重kg_变化'].cumsum()

        processed_sheets[sheet_name] = df

    # 将处理后的DataFrame写入新的Excel文件 'data3.xlsx'
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    st.success("已拓展“累积”特征")

@st.cache_data
def xd6():
    xls = pd.ExcelFile(file_out)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    n7 = len(columns)

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)

        if f'{df.columns[2]}_变化' in df.columns and f'{df.columns[1]}_变化' in df.columns:
            df['产物转化率'] = df[f'{df.columns[2]}_变化']/ df[f'{df.columns[1]}_变化']
        if '菌浓g/50mL_变化' in df.columns and f'{df.columns[1]}_变化' in df.columns:
            df['菌体g转化率'] = df['菌浓g/50mL_变化'] / df[f'{df.columns[1]}_变化']
        if '菌浓mL/50mL_变化' in df.columns and f'{df.columns[1]}_变化' in df.columns:
            df['菌体mL转化率'] = df['菌浓mL/50mL_变化'] / df[f'{df.columns[1]}_变化']

        processed_sheets[sheet_name] = df

    # 将处理后的DataFrame写入新的Excel文件 'data3.xlsx'
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    st.success("已拓展“转化率”特征")

@st.cache_data
def xd7():
    xls = pd.ExcelFile(file_out)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    n8 = len(columns)

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)
        if '菌浓mL/50mL' in df.columns and '发酵时间' in df.columns:
            df['比生长速率ml'] = np.log(df['菌浓mL/50mL'] / df['菌浓mL/50mL'].shift(1)) / df['发酵时间']
        if '菌浓g/50mL' in df.columns and '发酵时间' in df.columns:
            df['比生长速率g'] = np.log(df['菌浓g/50mL'] / df['菌浓g/50mL'].shift(1)) / df['发酵时间']

        processed_sheets[sheet_name] = df

    # 将处理后的DataFrame写入新的Excel文件 'data3.xlsx'
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    st.success("已拓展“生物学”特征")

@st.cache_data
def xd8():
    xls = pd.ExcelFile(file_out)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    n9 = len(columns)

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)

        df['下一产物'] = df[f'{df.columns[2]}'].shift(-1)
        df['下一底物'] = df[f'{df.columns[1]}'].shift(-1)

        processed_sheets[sheet_name] = df

    # 将处理后的DataFrame写入新的Excel文件 'data3.xlsx'
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

@st.cache_data
def xd9():
    def merge_sheets(input_file, output_file):
        try:
            # 1. 读取Excel文件
            xls = pd.ExcelFile(input_file)

            # 2. 初始化存储合并数据的列表
            all_data = []

            # 3. 遍历所有sheet
            for sheet_name in xls.sheet_names:
                # 读取当前sheet数据（第一行为列名，从第二行开始为数据）
                df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                # 添加到合并列表
                all_data.append(df)

            # 4. 合并所有数据
            merged_data = pd.concat(all_data, ignore_index=True)

            # 5. 保存结果
            merged_data.to_excel(output_file, index=False)
            st.success("特征拓展完成，结果已保存至：d1")

        except FileNotFoundError:
            st.error(f"错误：找不到输入文件 {input_file}")
        except Exception as e:
            st.error(f"处理过程中发生错误：{str(e)}")

    # 使用示例
    if __name__ == "__main__":
        input_path = file_out  # 输入文件路径
        output_path = file_out  # 输出文件路径
        merge_sheets(input_path, output_path)



#计算f值
@st.cache_data
def fscore1_a(pre):
    # 读取Excel文件
    df = pd.read_excel(file_out, engine='openpyxl')
    df = df[~df.isna().any(axis=1)]
    # 提取目标变量y（第三列）
    y = df[pre]  # 注意：列索引从0开始，第三列对应索引2

    # 提取特征X（从第五列到最后一列）
    X = df.iloc[:, :-2]

    # 标准化特征X（SVR 对特征缩放敏感）
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用SVR进行训练
    svr = SVR(kernel='linear')  # 使用线性核的SVR
    svr.fit(X_scaled, y)

    # 获取SVR模型中的特征系数，这些系数可以视作F-score
    feature_importance = svr.coef_[0]  # 获取特征系数

    # 对特征进行排名
    feature_ranking = sorted(zip(X.columns, feature_importance), key=lambda x: abs(x[1]), reverse=True)

    # 你也可以使用f_regression来计算F-score
    f_scores, _ = f_regression(X_scaled, y)

    # 输出通过f_regression计算的F-scores排名
    f_score_ranking = sorted(zip(X.columns, f_scores), key=lambda x: x[1], reverse=True)
    return f_score_ranking


    # 读取Excel文件中的所有sheet
@st.cache_data
def fscore_a(pre,lis1):
    EV=0
    lis = [item[0] for item in lis1]
    # 读取Excel文件中的所有sheet
    file_path = file_out
    data = pd.read_excel(file_path)  # 读取
    data = data.dropna(how='any')
        # 1
    feature = lis
    features = feature[:1]
    target = pre

        # 分割数据集
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
        # 初始化标准化器
    scaler_X = MinMaxScaler()
    scaler_y= MinMaxScaler()

        # 对训练集进行拟合和变换
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

        # 对测试集仅进行变换
    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

    X1 = X_train
    y1 = y_train

        # 创建SVR模型，使用RBF核函数
    regressor = SVR(kernel='linear')
    kf = KFold(n_splits=25, shuffle=True)
    scores = cross_val_score(regressor, X1, y1, cv=kf, scoring='neg_mean_squared_error')

    # 计算平均 MSE
    average_mse_1 = np.mean(-scores)

        # all
    feature = lis
    features = feature[:len(lis)]
    target = pre

        # 分割数据集
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
        # 初始化标准化器
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

        # 对训练集进行拟合和变换
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

        # 对测试集仅进行变换
    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

    X1 = X_train
    y1 = y_train

        # 创建SVR模型，使用RBF核函数
    regressor = SVR(kernel='linear')
    kf = KFold(n_splits=25, shuffle=True)
    scores = cross_val_score(regressor, X1, y1, cv=kf, scoring='neg_mean_squared_error')

        # 计算平均 MSE
    average_mse_all = np.mean(-scores)
    ev = 1
    mse_ev = average_mse_1
    EV = len(lis)
    mse_EV = average_mse_all

    def calculate_average(data, i, j):
            # 使用切片提取从第i到第j个元素
        if i <= j:
            sublist = data[i:j + 1]

                # 计算这些元素的总和
            total_sum = sum(sublist)

                # 计算元素的个数
            count = len(sublist)

                # 计算并返回平均值
            if count > 0:
                return total_sum / count
            else:
                return 0  # 如果没有元素，则返回0
        else:
            sublist = data[j:i + 1]

                # 计算这些元素的总和
            total_sum = sum(sublist)

                # 计算元素的个数
            count = len(sublist)

                # 计算并返回平均值
            if count > 0:
                return total_sum / count
            else:
                return 0  # 如果没有元素，则返回0

    def find_greater_than_m(data, m):
            # 使用列表推导式找到所有大于m的元素
        result = [x for x in data if x > m]
        return result
    #lis1(名称，分数) lis(名称) lis2(分数)
    lis2 = [item[1] for item in lis1]
    while abs(EV - ev) > 1:
        average = calculate_average(lis2, EV - 1, ev - 1)
        new_list = find_greater_than_m(lis2, average)
        n = len(new_list)

        feature = lis
        features = feature[:n]
        target = pre

            # 分割数据集
        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
            # 初始化标准化器
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

            # 对训练集进行拟合和变换
        X_train = scaler_X.fit_transform(X_train)
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

            # 对测试集仅进行变换
        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

        X1 = X_train
        y1 = y_train

            # 创建SVR模型，使用RBF核函数
        regressor = SVR(kernel='linear')
        kf = KFold(n_splits=25, shuffle=True)
        scores = cross_val_score(regressor, X1, y1, cv=kf, scoring='neg_mean_squared_error')

            # 计算平均 MSE
        average_mse = np.mean(-scores)

        if average_mse <= mse_EV:
            mse_EV = average_mse
            EV = n
        else:
            mse_ev = average_mse
            ev = n
    return EV


@st.cache_data
def fscore1_s(pre):
    # 读取Excel文件
    df = pd.read_excel(file_out, engine='openpyxl')
    df = df[~df.isna().any(axis=1)]
    # 提取目标变量y（第三列）
    y = df[pre]  # 注意：列索引从0开始，第三列对应索引2

    # 提取特征X（从第五列到最后一列）
    X = df.iloc[:, :-2]  # 第五列对应索引4，直到最后一列

    # 标准化特征X（SVR 对特征缩放敏感）
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用SVR进行训练
    svr = SVR(kernel='linear')  # 使用线性核的SVR
    svr.fit(X_scaled, y)

    # 获取SVR模型中的特征系数，这些系数可以视作F-score
    feature_importance = svr.coef_[0]  # 获取特征系数

    # 对特征进行排名
    feature_ranking = sorted(zip(X.columns, feature_importance), key=lambda x: abs(x[1]), reverse=True)

    # 你也可以使用f_regression来计算F-score
    f_scores, _ = f_regression(X_scaled, y)

    # 输出通过f_regression计算的F-scores排名
    f_score_ranking = sorted(zip(X.columns, f_scores), key=lambda x: x[1], reverse=True)
    return f_score_ranking


    # 读取Excel文件中的所有sheet
@st.cache_data
def fscore_s(pre,lis1):
    EV=0
    lis = [item[0] for item in lis1]
        # 读取Excel文件中的所有sheet
    file_path = file_out
    data = pd.read_excel(file_path)  # 读取
    data = data.dropna(how='any')
        # 1
    feature = lis
    features = feature[:1]
    target = pre

        # 分割数据集
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
        # 初始化标准化器
    scaler_X = MinMaxScaler()
    scaler_y= MinMaxScaler()

        # 对训练集进行拟合和变换
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

        # 对测试集仅进行变换
    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

    X1 = X_train
    y1 = y_train

        # 创建SVR模型，使用RBF核函数
    regressor = SVR(kernel='linear')
    kf = KFold(n_splits=25, shuffle=True)
    scores = cross_val_score(regressor, X1, y1, cv=kf, scoring='neg_mean_squared_error')

        # 计算平均 MSE
    average_mse_1 = np.mean(-scores)

        # all
    feature = lis
    features = feature[:len(lis)]
    target = pre

        # 分割数据集
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
        # 初始化标准化器
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

        # 对训练集进行拟合和变换
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

        # 对测试集仅进行变换
    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

    X1 = X_train
    y1 = y_train

        # 创建SVR模型，使用RBF核函数
    regressor = SVR(kernel='linear')
    kf = KFold(n_splits=25, shuffle=True)
    scores = cross_val_score(regressor, X1, y1, cv=kf, scoring='neg_mean_squared_error')

        # 计算平均 MSE
    average_mse_all = np.mean(-scores)
    ev = 1
    mse_ev = average_mse_1
    EV = len(lis)
    mse_EV = average_mse_all

    def calculate_average(data, i, j):
            # 使用切片提取从第i到第j个元素
        if i <= j:
            sublist = data[i:j + 1]

                # 计算这些元素的总和
            total_sum = sum(sublist)

                # 计算元素的个数
            count = len(sublist)

                # 计算并返回平均值
            if count > 0:
                return total_sum / count
            else:
                return 0  # 如果没有元素，则返回0
        else:
            sublist = data[j:i + 1]

                # 计算这些元素的总和
            total_sum = sum(sublist)

                # 计算元素的个数
            count = len(sublist)

                # 计算并返回平均值
            if count > 0:
                return total_sum / count
            else:
                return 0  # 如果没有元素，则返回0

    def find_greater_than_m(data, m):
            # 使用列表推导式找到所有大于m的元素
        result = [x for x in data if x > m]
        return result
    #lis1(名称，分数) lis(名称) lis2(分数)
    lis2 = [item[1] for item in lis1]
    while abs(EV - ev) > 1:
        average = calculate_average(lis2, EV - 1, ev - 1)
        new_list = find_greater_than_m(lis2, average)
        n = len(new_list)

        feature = lis
        features = feature[:n]
        target = pre

            # 分割数据集
        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
            # 初始化标准化器
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

            # 对训练集进行拟合和变换
        X_train = scaler_X.fit_transform(X_train)
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

            # 对测试集仅进行变换
        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

        X1 = X_train
        y1 = y_train

            # 创建SVR模型，使用RBF核函数
        regressor = SVR(kernel='linear')
        kf = KFold(n_splits=25, shuffle=True)
        scores = cross_val_score(regressor, X1, y1, cv=kf, scoring='neg_mean_squared_error')

            # 计算平均 MSE
        average_mse = np.mean(-scores)

        if average_mse <= mse_EV:
            mse_EV = average_mse
            EV = n
        else:
            mse_ev = average_mse
            ev = n
    return EV



@st.cache_data
def svrm_a(pre,lis):
    file_path = file_out
    data = pd.read_excel(file_path)  # 读取
    data = data.dropna(how='any')
    # SVR
    r2 = 0
    features = lis
    target = pre

        # 分割数据集
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
        # 初始化标准化器
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

        # 对训练集进行拟合和变换
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

        # 对测试集仅进行变换
    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

    X1 = X_train
    y1 = y_train

        # 定义SVR模型的超参数搜索空间
    search_space = {
            'C': (1e-3, 1e+0, 'log-uniform'),  # 惩罚项参数 C
            'epsilon': (0.01, 0.1),  # 损失函数的 epsilon
            'kernel': ['linear', 'rbf'],  # 核函数类型
            'gamma': ['scale', 'auto'],  # 核函数的系数
        }

        # 使用贝叶斯优化进行超参数搜索
    opt = BayesSearchCV(
            SVR(),  # 模型
            search_space,  # 超参数空间
            n_iter=25,  # 迭代次数
            cv=5,  # 交叉验证次数
        )

        # 拟合模型
    opt.fit(X1, y1)

        # 创建SVR模型，使用RBF核函数
    regressor_svr_a = SVR(C=opt.best_params_['C'], epsilon=opt.best_params_['epsilon'],
                            gamma=opt.best_params_['gamma'], kernel=opt.best_params_['kernel'])

    scores = cross_val_score(regressor_svr_a, X1, y1, cv=25, scoring='neg_mean_squared_error')
    scores_mae = cross_val_score(regressor_svr_a, X1, y1, cv=25, scoring='neg_mean_absolute_error')
    scores_r2 = cross_val_score(regressor_svr_a, X1, y1, cv=25, scoring='r2')
    # 计算平均 MSE
    mse = np.mean(-scores)
    rmse = np.sqrt(mse)
    mae = np.mean(-scores_mae)
    r2 = np.mean(scores_r2)
        #st.markdown(r2)
    # 输出结果
    st.markdown("SVR模型性能指标：")
    st.markdown(f"MSE: {mse}")
    st.markdown(f"RMSE: {rmse}")
    st.markdown(f"MAE: {mae}")
    st.markdown(f"R²: {r2}")
    regressor_svr_a.fit(X1, y1)
    return regressor_svr_a,mse,scaler_X,scaler_y

@st.cache_data
def svrm_s(pre,lis):
    file_path = file_out
    data = pd.read_excel(file_path)  # 读取
    data = data.dropna(how='any')
    # SVR
    r2 = 0
    features = lis
    target = pre

        # 分割数据集
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
        # 初始化标准化器
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

        # 对训练集进行拟合和变换
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

        # 对测试集仅进行变换
    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

    X1 = X_train
    y1 = y_train

        # 定义SVR模型的超参数搜索空间
    search_space = {
            'C': (1e-3, 1e+0, 'log-uniform'),  # 惩罚项参数 C
            'epsilon': (0.01, 0.1),  # 损失函数的 epsilon
            'kernel': ['linear', 'rbf'],  # 核函数类型
            'gamma': ['scale', 'auto'],  # 核函数的系数
        }

        # 使用贝叶斯优化进行超参数搜索
    opt = BayesSearchCV(
            SVR(),  # 模型
            search_space,  # 超参数空间
            n_iter=25,  # 迭代次数
            cv=5,  # 交叉验证次数
        )

        # 拟合模型
    opt.fit(X1, y1)

        # 创建SVR模型，使用RBF核函数
    regressor_svr_s = SVR(C=opt.best_params_['C'], epsilon=opt.best_params_['epsilon'],
                            gamma=opt.best_params_['gamma'], kernel=opt.best_params_['kernel'])


    scores = cross_val_score(regressor_svr_s, X1, y1, cv=25, scoring='neg_mean_squared_error')
    scores_mae = cross_val_score(regressor_svr_s, X1, y1, cv=25, scoring='neg_mean_absolute_error')
    scores_r2 = cross_val_score(regressor_svr_s, X1, y1, cv=25, scoring='r2')
        # 计算平均 MSE
    mse = np.mean(-scores)
    rmse = np.sqrt(mse)
    mae = np.mean(-scores_mae)
    r2 = np.mean(scores_r2)
    # 输出结果
    st.markdown("SVR模型性能指标：")
    st.markdown(f"MSE: {mse}")
    st.markdown(f"RMSE: {rmse}")
    st.markdown(f"MAE: {mae}")
    st.markdown(f"R²: {r2}")
    regressor_svr_s.fit(X1,y1)
    return regressor_svr_s,mse,scaler_X,scaler_y

@st.cache_data
def plssvrm_a(pre,lis):
    file_path = file_out
    data = pd.read_excel(file_path)  # 读取
    data = data.dropna(how='any')
    # SVR
    r2 = 0
    features = lis
    target = pre

        # 分割数据集
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
        # 初始化标准化器
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

        # 对训练集进行拟合和变换
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

        # 对测试集仅进行变换
    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

        # 使用 PLS 进行特征提取，选择提取2个主成分
    pls = PLSRegression(n_components=5)

        # 拟合 PLS 模型
    pls.fit(X_train, y_train)

        # 提取特征，得到新的特征矩阵 X_pls（经过降维后的数据）
    X_pls = pls.transform(X_train)
    X_test_pls = pls.transform(X_test)
    # 提取特征，得到新的特征矩阵 X_pls（经过降维后的数据）
    X_test_pls = pls.transform(X_test)

    X1 = X_pls
    y1 = y_train

        # 定义SVR模型的超参数搜索空间
    search_space = {
            'C': (1e-3, 1e+0, 'log-uniform'),  # 惩罚项参数 C
            'epsilon': (0.01, 0.1),  # 损失函数的 epsilon
            'kernel': ['linear', 'rbf'],  # 核函数类型
            'gamma': ['scale', 'auto'],  # 核函数的系数
        }

        # 使用贝叶斯优化进行超参数搜索
    opt = BayesSearchCV(
            SVR(),  # 模型
            search_space,  # 超参数空间
            n_iter=25,  # 迭代次数
            cv=5,  # 交叉验证次数
        )

        # 拟合模型
    opt.fit(X1, y1)

        # 创建SVR模型，使用RBF核函数
    regressor_plssvr_a = SVR(C=opt.best_params_['C'], epsilon=opt.best_params_['epsilon'],
                            gamma=opt.best_params_['gamma'], kernel=opt.best_params_['kernel'])

    scores = cross_val_score(regressor_plssvr_a, X1, y1, cv=25, scoring='neg_mean_squared_error')
    scores_mae = cross_val_score(regressor_plssvr_a, X1, y1, cv=25, scoring='neg_mean_absolute_error')
    scores_r2 = cross_val_score(regressor_plssvr_a, X1, y1, cv=25, scoring='r2')
    # 计算平均 MSE
    mse = np.mean(-scores)
    rmse = np.sqrt(mse)
    mae = np.mean(-scores_mae)
    r2 = np.mean(scores_r2)
    # 输出结果
    st.markdown("PLS-SVR模型性能指标：")
    st.markdown(f"MSE: {mse}")
    st.markdown(f"RMSE: {rmse}")
    st.markdown(f"MAE: {mae}")
    st.markdown(f"R²: {r2}")
    regressor_plssvr_a.fit(X1, y1)
    return regressor_plssvr_a,mse,scaler_X,scaler_y,pls.transform

@st.cache_data
def plssvrm_s(pre,lis):
    file_path = file_out
    data = pd.read_excel(file_path)  # 读取
    data = data.dropna(how='any')
    # SVR
    r2 = 0
    features = lis
    target = pre

        # 分割数据集
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
        # 初始化标准化器
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

        # 对训练集进行拟合和变换
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

        # 对测试集仅进行变换
    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

        # 使用 PLS 进行特征提取，选择提取2个主成分
    pls = PLSRegression(n_components=4)

        # 拟合 PLS 模型
    pls.fit(X_train, y_train)

        # 提取特征，得到新的特征矩阵 X_pls（经过降维后的数据）
    X_pls = pls.transform(X_train)
    X_test_pls = pls.transform(X_test)
        # 提取特征，得到新的特征矩阵 X_pls（经过降维后的数据）
    X_test_pls = pls.transform(X_test)

    X1 = X_pls
    y1 = y_train

        # 定义SVR模型的超参数搜索空间
    search_space = {
            'C': (1e-3, 1e+0, 'log-uniform'),  # 惩罚项参数 C
            'epsilon': (0.01, 0.1),  # 损失函数的 epsilon
            'kernel': ['linear', 'rbf'],  # 核函数类型
            'gamma': ['scale', 'auto'],  # 核函数的系数
        }

        # 使用贝叶斯优化进行超参数搜索
    opt = BayesSearchCV(
            SVR(),  # 模型
            search_space,  # 超参数空间
            n_iter=25,  # 迭代次数
            cv=5,  # 交叉验证次数
        )

        # 拟合模型
    opt.fit(X1, y1)

        # 创建SVR模型，使用RBF核函数
    regressor_plssvr_s = SVR(C=opt.best_params_['C'], epsilon=opt.best_params_['epsilon'],
                            gamma=opt.best_params_['gamma'], kernel=opt.best_params_['kernel'])

    scores = cross_val_score(regressor_plssvr_s, X1, y1, cv=25, scoring='neg_mean_squared_error')
    scores_mae = cross_val_score(regressor_plssvr_s, X1, y1, cv=25, scoring='neg_mean_absolute_error')
    scores_r2 = cross_val_score(regressor_plssvr_s, X1, y1, cv=25, scoring='r2')
    # 计算平均 MSE
    mse = np.mean(-scores)
    rmse = np.sqrt(mse)
    mae = np.mean(-scores_mae)
    r2 = np.mean(scores_r2)
    # 输出结果
    st.markdown("PLS-SVR模型性能指标：")
    st.markdown(f"MSE: {mse}")
    st.markdown(f"RMSE: {rmse}")
    st.markdown(f"MAE: {mae}")
    st.markdown(f"R²: {r2}")
    regressor_plssvr_s.fit(X1, y1)
    return regressor_plssvr_s,mse,scaler_X,scaler_y,pls.transform


#预测
#@st.cache_data
def xd00():
    excel_file = pd.ExcelFile(uploaded_file1, engine="openpyxl")
    xls = pd.ExcelFile(excel_file)
    df = pd.read_excel(xls)
    columns = df.columns.tolist()
    m1 = len(columns)+1
    # 创建ExcelWriter对象以保存结果
    with pd.ExcelWriter(file_out_in) as writer:
        # 遍历每个sheet
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0, index_col=None)
            change_columns = {
                '发酵时间': columns[0]
            }
            for new_col, base_col in change_columns.items():
                df[new_col] = df[base_col].diff()
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return m1

#@st.cache_data
def xd11():
    xls = pd.ExcelFile(file_out_in)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)

        # 获取第二列到第十二列的索引 (索引从0开始，因此第二列是索引1，第十二列是索引11)
        columns_to_process = df.columns[1:m1 - 1]

        # 计算差值并添加到DataFrame中
        for col in columns_to_process:
            # 计算差值，去掉第一行，因为没有上一行可以计算差值
            df[f'{col}_变化'] = df[col].diff()

        # 将处理后的DataFrame存储在字典中
        processed_sheets[sheet_name] = df

    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
#@st.cache_data
def xd22():
    xls = pd.ExcelFile(file_out_in)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    n3 = len(columns)

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)
        for col in range(n1, n3):  # 13对应的是第14列，因为索引从0开始
            df[f'变化率_{df.columns[col]}'] = df.iloc[:, col] / df['发酵时间']  # 12对应的是第13列

        processed_sheets[sheet_name] = df

    # 将处理后的DataFrame写入新的Excel文件 'data3.xlsx'
    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

#@st.cache_data
def xd33():
    xls = pd.ExcelFile(file_out_in)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    n4 = len(columns)

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)
        for col in range(1, n4):  # 第2列的索引是1，第35列的索引是34
            # 新列的值为上一行的数据
            df[f'上一_{df.columns[col]}'] = df.iloc[:, col].shift(1)

        processed_sheets[sheet_name] = df

    # 将处理后的DataFrame写入新的Excel文件 'data3.xlsx'
    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

#@st.cache_data
def xd44():
    xls = pd.ExcelFile(file_out_in)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    n5 = len(columns)

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)

        for col in range(1, n5):  # 索引从1开始，第二列是索引1，直到第69列索引68
            new_col_name = f"{df.columns[col]}_squared"  # 新特征列名称
            df[new_col_name] = df.iloc[:, col] ** 2  # 计算平方并添加新列

        processed_sheets[sheet_name] = df

    # 将处理后的DataFrame写入新的Excel文件 'data3.xlsx'
    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

#@st.cache_data
def xd55():
    xls = pd.ExcelFile(file_out_in)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    n6 = len(columns)

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)

        if f'{df.columns[1]}_变化' in df.columns:
            df['耗糖累积量'] = df[f'{df.columns[1]}_变化'].cumsum()
        if '碱重kg_变化' in df.columns:
            df['耗碱累积量'] = df['碱重kg_变化'].cumsum()

        processed_sheets[sheet_name] = df

    # 将处理后的DataFrame写入新的Excel文件 'data3.xlsx'
    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

#@st.cache_data
def xd66():
    xls = pd.ExcelFile(file_out_in)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    n7 = len(columns)

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)

        if f'{df.columns[2]}_变化' in df.columns and f'{df.columns[1]}_变化' in df.columns:
            df['产物转化率'] = df[f'{df.columns[2]}_变化']/ df[f'{df.columns[1]}_变化']
        if '菌浓g/50mL_变化' in df.columns and f'{df.columns[1]}_变化' in df.columns:
            df['菌体g转化率'] = df['菌浓g/50mL_变化'] / df[f'{df.columns[1]}_变化']
        if '菌浓mL/50mL_变化' in df.columns and f'{df.columns[1]}_变化' in df.columns:
            df['菌体ml转化率'] = df['菌浓mL/50mL_变化'] / df[f'{df.columns[1]}_变化']

        processed_sheets[sheet_name] = df

    # 将处理后的DataFrame写入新的Excel文件 'data3.xlsx'
    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

#@st.cache_data
def xd77():
    xls = pd.ExcelFile(file_out_in)
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}

    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    n8 = len(columns)

    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)
        if '菌浓mL/50mL' in df.columns and '发酵时间' in df.columns:
            df['比生长速率ml'] = np.log(df['菌浓mL/50mL'] / df['菌浓mL/50mL'].shift(1)) / df['发酵时间']
        if '菌浓g/50mL' in df.columns and '发酵时间' in df.columns:
            df['比生长速率g'] = np.log(df['菌浓g/50mL'] / df['菌浓g/50mL'].shift(1)) / df['发酵时间']

        processed_sheets[sheet_name] = df

    # 将处理后的DataFrame写入新的Excel文件 'data3.xlsx'
    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


st.title("基于SVR的生物过程预测")
# 创建一个按钮
if st.button('说明书'):
    # 当按钮被点击时，显示一个带有换行的文字内容
    with st.expander("基于SVR的生物过程预测使用说明书"):
        st.markdown("1. 系统概述")
        st.markdown("该网站主要分为两个部分：生物过程预测，模型预测控制（可选）")
        st.markdown("2. 生物过程预测")
        st.markdown("生物过程预测包含5个模块")
        st.markdown("2.1 文件读取")
        st.markdown("功能：读取选择的包含训练数据的Excel文件。")
        st.markdown("注意：文件应包含多个sheet（每个sheet表示一个批次），每个sheet的格式如下：横轴为测量点，纵轴为特征名称，特征名称顺序为：发酵周期/h、底物、产物、其他特征1、其他特征2...（前三个特征不可改变，菌浓相关特征应命名为“菌浓ml”或“菌浓g”）。测量时间间隔应尽可能相等。")
        st.markdown("操作：点击“文件读取”按钮以读取训练数据，随后可点击“文件预览”按钮以浏览读取的训练数据。")
        st.markdown("2.2 特征拓展")
        st.markdown("功能：对已有的特征进行拓展，丰富数据特征，更好的学习其中的规律")
        st.markdown("建议：建议全选所有特征进行拓展。")
        st.markdown("操作：多选框选择拓展类型，点击“开始拓展”按钮进行特征拓展，拓展完成后，数据将保存为名为“d1”的Excel文件。点击“处理数据预览”按钮以预览拓展后的数据。")
        st.markdown("2.3 特征抽提")
        st.markdown("功能：对产物预测模型与底物预测模型进行特征抽提。")
        st.markdown("操作：点击“开始特征抽提”按钮，系统将开始特征抽提过程。完成后，会展示产物预测模型与底物预测模型所抽提的特征。")
        st.markdown("提示：此过程可能需要较长时间，请耐心等待。")
        st.markdown("2.4 模型训练")
        st.markdown("功能：训练不同类型的SVR模型，包括产物预测模型与底物预测模型。")
        st.markdown("操作：点击“开始训练产物预测模型”与“开始训练底物预测模型”按钮，系统将分别训练不同模型。训练完成后，系统将展示各个模型的性能指标。")
        st.markdown("提示：此过程可能需要较长时间，请耐心等待。MSE、RMSE、MAE越低，R方越高，模型性能越好。")
        st.markdown("2.5 预测")
        st.markdown("功能：预测待测数据的产物生成速率与下一个测量点的产物浓度。")
        st.markdown("操作：")
        st.markdown("①上传待预测的数据。确保数据的特征列与训练数据完全相同，且至少包含三个测量点。")
        st.markdown("②点击“输入数据预处理”按钮，数据将根据“特征拓展”中的设置进行相同类型的拓展。")
        st.markdown("③预处理完成后，展示处理后的待预测数据。")
        st.markdown("④选择“开始预测产物”或“开始预测底物”按钮进行预测。选择一个训练好的模型进行预测（推荐选择MSE较低的模型）。")
        st.markdown("3. 模型预测控制（可选）")
        st.markdown("操作：")
        st.markdown("①在复选框中选择需要进行优化的控制变量。建议选择的控制变量不超过3个。")
        st.markdown("②点击“输入上下限按钮”，设置各控制变量的下限与上限。控制变量的取值需符合实际情况。")
        st.markdown("③点击“开始”按钮，开始进行模型预测控制。")
        st.markdown("4. 缓存清理")
        st.markdown("若需要清理缓存，请点击网页最下方的“清理缓存”按钮。")
        st.markdown("5. 注意事项")
        st.markdown("在进行模型训练、特征抽提等需要较长时间的操作时，请耐心等待，避免频繁操作导致系统不稳定")
        st.markdown("本系统仅适用于符合规定格式的训练数据，上传的数据应严格按照说明书中要求的格式进行整理")


st.header("1.文件读取")
# 文件上传
uploaded_file = st.file_uploader("选择一个 Excel 文件", type=["xlsx", "xls"],key="file_uploader_0")
# 如果用户上传了文件
if uploaded_file is not None:
    #文件读取
    excel_file = pd.ExcelFile(uploaded_file, engine="openpyxl")
    xls = pd.ExcelFile(excel_file)
    df1 = pd.read_excel(xls)
    # 文件预览
    # 创建按钮
    if st.button('文件预览'):
        st.session_state.a = 'a'
    # 显示结果
    if 'a' in st.session_state:
        st.subheader('数据预览')
        st.write(df1.head())
    #添加发酵时间
    n1=xd0()
    mpc_lis = df1.columns
    options_mpc = st.multiselect(
        label='请选择需要优化的控制变量',
        options=mpc_lis,
        default=None,
        format_func=str,
        help='请选择拓展特征类型'
    )
    kk.append(1)



st.header("2.特征拓展")
if 1 in kk:
    options = st.multiselect(
        label='请选择拓展特征类型',
        options=('变化特征', '变化率特征', '时序特征', '多项式特征', '累积特征', '转化率特征', '生物学特征'),
        default=None,
        format_func=str,
        help='请选择拓展特征类型'
    )
    if st.button('开始拓展'):
        st.session_state.b = 'b'

    if 'b' in st.session_state:
        if '变化特征' in options :
            xd1()
        if '变化率特征' in options :
            xd2()
        if '时序特征' in options :
            xd3()
        if '多项式特征' in options :
            xd4()
        if '累积特征' in options :
            xd5()
        if '转化率特征' in options :
            xd6()
        if '生物学特征' in options :
            xd7()
        xd8()
        xd9()
        excel_file = pd.ExcelFile(uploaded_file, engine="openpyxl")
        xls = pd.ExcelFile(file_out)
        df2 = pd.read_excel(xls)
        if st.button('处理数据预览'):
            st.session_state.aa = 'aa'
        # 显示结果
        if 'aa' in st.session_state:
            st.subheader('数据预览')
            st.write(df2.head(20))
        pre1=f'{df2.columns[-2]}'#酸钠
        pre2=f'{df2.columns[-1]}'#残糖
        kk.append(2)



st.header("3.特征抽提")
if 2 in kk:
    if st.button('开始特征抽提'):
        st.session_state.c = 'c'
    if 'c' in st.session_state:
        #酸钠
        st.write('预测产物特征抽提:')
        aflist = fscore1_a(pre1)
        aEV = fscore_a(pre1, aflist)
        st.write('选择特征数：',aEV)
        acid_con = list(dict.fromkeys([item[0] for item in aflist][:aEV] + options_mpc))
        st.write(acid_con)
        #残糖
        st.write('预测底物特征抽提：')
        sflist = fscore1_s(pre2)
        sEV = fscore_s(pre2, sflist)
        st.write('选择特征数：',sEV)
        sur_con = list(dict.fromkeys([item[0] for item in sflist][:sEV] + options_mpc))
        st.write(sur_con)
        kk.append(3)



st.header("4.模型训练")
if 3 in kk:
    if st.button('开始训练产物预测模型'):
        st.session_state.d = 'd'
    if 'd' in st.session_state:
        svrmse_aa = '未预测'
        plssvrmse_aa = '未预测'
        bi2svrmse_aa = '尽请期待'
        svrm_aa, svrmse_aa,sacX,sacy = svrm_a(pre1, acid_con)
        plssvrm_aa, plssvrmse_aa,psacX,psacy,ps_aa = plssvrm_a(pre1, acid_con)
        svrmse_aa=round(svrmse_aa,4)
        plssvrmse_aa=round(plssvrmse_aa,4)
        kk.append(4)

    if st.button('开始训练底物预测模型'):
        st.session_state.e = 'e'
    if 'e' in st.session_state:
        svrmse_ss = '未预测'
        plssvrmse_ss = '未预测'
        bi2svrmse_ss = '尽请期待'
        svrm_ss, svrmse_ss ,sscX,sscy= svrm_s(pre2, sur_con)
        plssvrm_ss, plssvrmse_ss ,psscX,psscy,ps_ss= plssvrm_s(pre2, sur_con)
        svrmse_ss = round(svrmse_ss, 4)
        plssvrmse_ss = round(plssvrmse_ss, 4)
        kk.append(4)



st.header("5.预测")
if 4 in kk:
    # 文件上传
    uploaded_file1 = st.file_uploader("选择一个 Excel 文件", type=["xlsx", "xls"],key="file_uploader_1")
    # 如果用户上传了文件
    if uploaded_file1 is not None:
        # 文件读取
        excel_file1 = pd.ExcelFile(uploaded_file1, engine="openpyxl")
        xls1 = pd.ExcelFile(excel_file1)
        df11 = pd.read_excel(xls1)
        perld = st.number_input("输入发酵周期")
        # 文件预览
        # 创建按钮
        if st.button('输入数据预处理'):
            st.session_state.f = 'f'
        # 显示结果
        if 'f' in st.session_state:
            m1 = xd00()
            if '变化特征' in options :
                xd11()
            if '变化率特征' in options :
                xd22()
            if '时序特征' in options :
                xd33()
            if '多项式特征' in options :
                xd44()
            if '累积特征' in options :
                xd55()
            if '转化率特征' in options :
                xd66()
            if '生物学特征' in options :
                xd77()
            excel_file2 = pd.ExcelFile(file_out_in, engine="openpyxl")
            xls2 = pd.ExcelFile(excel_file2)
            df22 = pd.read_excel(xls2)
            prdata = df22.iloc[-1]
            st.write(prdata)
            kk.append(5)



if 5 in kk:
    if st.button('开始预测产物'):
        st.session_state.g = 'g'
    if 'g' in st.session_state:
        model = st.radio(
            label='请选择产物预测模型',
            options=(f'SVR   MSE:{svrmse_aa}', f'PLS-SVR   MSE:{plssvrmse_aa}', f'BI-2SVR   MSE:{bi2svrmse_aa}'),
            index=2,
            format_func=str,
            help='推荐使用MSE较小的模型'
        )
        if model == f'SVR   MSE:{svrmse_aa}':
            features = acid_con
            X = df22[features]
            predata1 = sacX.transform(X)
            y_preddata1_a = svrm_aa.predict(predata1[-1].reshape(1,-1))
            data_original = sacy.inverse_transform(y_preddata1_a.reshape(1,-1))
            st.write(f'底物消耗速率为{(data_original[0]-prdata[1])/prdata[m1-1]}')
            st.write(f'{perld}后{df22.columns[2]}预计浓度为{data_original[0]}')

        elif model ==f'PLS-SVR   MSE:{plssvrmse_aa}':
            features = acid_con
            X = df22[features]
            predata1 = psacX.transform(X)
            predata1 = ps_aa(predata1[-1].reshape(1,-1))
            y_preddata1_a = plssvrm_aa.predict(predata1)
            data_original = psacy.inverse_transform(y_preddata1_a.reshape(1,-1))
            st.write(f'底物消耗速率为{(data_original[0]-prdata[1])/prdata[m1-1]}')
            st.write(f'{perld}后{df22.columns[2]}预计浓度为{data_original[0]}')
        kk.append(6)

    if st.button('开始预测底物'):
        st.session_state.h = 'h'
    if 'h' in st.session_state:
        model = st.radio(
            label='请选择底物预测模型',
            options=(f'SVR   MSE:{svrmse_ss}', f'PLS-SVR   MSE:{plssvrmse_ss}', f'BI-2SVR   MSE:{bi2svrmse_ss}'),
            index=2,
            format_func=str,
            help='推荐使用MSE较小的模型'
        )
        if model == f'SVR   MSE:{svrmse_ss}':
            features = sur_con
            X = df22[features]
            predata2 = sscX.transform(X)
            y_preddata2_s = svrm_ss.predict(predata2[-1].reshape(1,-1))
            data_original_s = sscy.inverse_transform(y_preddata2_s.reshape(1,-1))
            st.write(f'底物消耗速率为{(data_original_s[0]-prdata[1])/prdata[m1-1]}')
            st.write(f'{perld}后{df22.columns[1]}预计浓度为{data_original_s[0]}')

        elif model ==f'PLS-SVR   MSE:{plssvrmse_ss}':
            features = sur_con
            X = df22[features]
            predata2 = psscX.transform(X)
            predata2 = ps_ss(predata2[-1].reshape(1,-1))
            y_preddata2_s = plssvrm_ss.predict(predata2)
            data_original_s = psscy.inverse_transform(y_preddata2_s.reshape(1,-1))
            st.write(f'底物消耗速率为{(data_original_s[0]-prdata[1])/prdata[m1-1]}')
            st.write(f'{perld}后{df22.columns[1]}预计浓度为{data_original_s[0]}')
        kk.append(6)



st.header("6.模型预测控制")
if 6 in kk:
    if st.button('输入上下限'):
        st.session_state.i = 'i'
    if 'i' in st.session_state:
        # 约束与边界
        bounds = []
        # 输入上下限
        for i in range(len(options_mpc)):
            st.write(f'请输入{options_mpc[i]}上下限')
            input_value_min = st.number_input("请输入下限", key=f"input_min_{i}")
            input_value_max = st.number_input("请输入上限", key=f"input_max_{i}")

            bounds.append((input_value_min, input_value_max))

    if st.button('开始'):
        st.session_state.j = 'j'
    if 'j' in st.session_state:
        features = acid_con
        mpc_x = df22.iloc[-1][features]
        state_params = mpc_x.to_numpy()
        initial_control_params = mpc_x[options_mpc].to_numpy()
        indices = [acid_con.index(x) for x in options_mpc]
        def objective(control_params):
            # 将控制参数与状态参数组合
            for i in indices:
                state_params[i] = control_params[indices.index(i)]
            input_features = sacX.transform(state_params.reshape(1, -1))
            # 使用SVR模型进行预测
            pre_mpc = svrm_aa.predict(input_features)
            prediction = sacy.inverse_transform(pre_mpc.reshape(1, 1))
            # 目标是最大化预测值，因此返回负值（minimize）
            return -prediction[0]

        # 调用优化函数
        result = minimize(objective, initial_control_params, bounds=bounds, method="Nelder-Mead",options={'maxiter': 1000, 'disp': True})
        # 输出结果
        optimal_control_params = result.x
        max_prediction = -result.fun  # 因为我们返回的是负值，反转回来

       # max_prediction = sacy.inverse_transform(np.array([max_prediction]).reshape(1, 1))

        for i in range(len(options_mpc)):
            st.write(f"{options_mpc[i]}最优控制参数: {optimal_control_params[i]}")
        st.write(f"最大预测值: {max_prediction}")

# 清除缓存

st.header("7.缓存清理")
if st.button('清理缓存'):
    st.cache_data.clear()
    st.write("缓存已清理！")







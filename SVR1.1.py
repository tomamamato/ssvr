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
from sklearn.metrics import r2_score, make_scorer
file_out="/tmp/d3.xlsx"#训练数据预处理后保存地址
file_out_in='/tmp/d4.xlsx'#预测数据预处理后保存地址
kf1=25#特征抽提时SVR交叉验证折数
kf2=2#模型训练时SVR交叉验证折数
pls_a_n=5
pls_s_n=7
pre1='下一产物'#预测目标
pre2='下一底物'#预测目标
options_mpc_0=[]
@st.cache_data
def xd0():#添加发酵时间
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
def xd1():#添加变化特征
    xls = pd.ExcelFile(file_out, engine="openpyxl")
    df = pd.read_excel(xls)
    # 新建一个字典存储处理后的每个sheet
    processed_sheets = {}
    # 获取当前sheet的所有列名
    columns = df.columns.tolist()
    # 遍历每个sheet
    for sheet_name in xls.sheet_names:
        # 读取每个sheet
        df = xls.parse(sheet_name)
        columns_to_process = df.columns[1:n1 - 1]
        # 计算差值并添加到DataFrame中
        for col in columns_to_process:
            df[f'{col}_变化'] = df[col].diff()
        # 将处理后的DataFrame存储在字典中
        processed_sheets[sheet_name] = df
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    st.success("已拓展“变化”特征")

@st.cache_data
def xd2():#添加变化率特征
    xls = pd.ExcelFile(file_out, engine="openpyxl")
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
        for col in range(n1, n3):
            df[f'变化率_{df.columns[col]}'] = df.iloc[:, col] / df['发酵时间']
        processed_sheets[sheet_name] = df
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    st.success("已拓展“变化率”特征")

@st.cache_data
def xd3():#添加时序特征
    xls = pd.ExcelFile(file_out, engine="openpyxl")
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
        for col in range(1, n4):
            # 新列的值为上一行的数据
            df[f'上一_{df.columns[col]}'] = df.iloc[:, col].shift(1)
        processed_sheets[sheet_name] = df
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    st.success("已拓展“时序”特征")

@st.cache_data
def xd4():#添加多项式特征
    xls = pd.ExcelFile(file_out, engine="openpyxl")
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
        for col in range(1, n5):
            new_col_name = f"{df.columns[col]}_squared"  # 新特征列名称
            df[new_col_name] = df.iloc[:, col] ** 2  # 计算平方并添加新列
        processed_sheets[sheet_name] = df
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    st.success("已拓展“多项式”特征")

@st.cache_data
def xd5():#添加累积特征
    xls = pd.ExcelFile(file_out, engine="openpyxl")
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
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    st.success("已拓展“累积”特征")

@st.cache_data
def xd6():#添加转化率特征
    xls = pd.ExcelFile(file_out, engine="openpyxl")
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
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

@st.cache_data
def xd7():#添加生物学特征
    xls = pd.ExcelFile(file_out, engine="openpyxl")
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
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    st.success("已拓展“生物学”特征")

@st.cache_data
def xd8():#添加预测目标
    xls = pd.ExcelFile(file_out, engine="openpyxl")
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
    with pd.ExcelWriter(file_out) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

@st.cache_data
def xd9():#多个sheet合并
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
            st.success("特征拓展完成，结果已保存至：d3")
        except FileNotFoundError:
            st.error(f"错误：找不到输入文件 {input_file}")
        except Exception as e:
            st.error(f"处理过程中发生错误：{str(e)}")
    if __name__ == "__main__":
        input_path = file_out  # 输入文件路径
        output_path = file_out  # 输出文件路径
        merge_sheets(input_path, output_path)

@st.cache_data
def fscore1_a(pre):#产物预测：计算f-score值
    # 读取Excel文件
    df = pd.read_excel(file_out, engine='openpyxl')
    df = df[~df.isna().any(axis=1)]
    # 提取目标变量y
    y = df[pre]
    # 提取特征X（从第一列到倒数第三列）
    X = df.iloc[:, :-2]
    # 标准化特征X
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

@st.cache_data
def fscore_a(pre,lis1):#产物预测：二分法特征抽提
    lis = [item[0] for item in lis1]
    # 读取Excel文件中的所有sheet
    file_path = file_out
    data = pd.read_excel(file_path)  # 读取
    data = data.dropna(how='any')#删除包含空值样本点
    #最高分数特征
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

    # 创建SVR模型，使用线性核函数
    regressor = SVR(kernel='linear',C=1,epsilon=0.01)
    kf = KFold(n_splits=kf1, shuffle=True)#交叉验证折数
    scores = cross_val_score(regressor, X1, y1, cv=kf, scoring='neg_mean_squared_error')

    # 计算平均 MSE
    average_mse_1 = np.mean(-scores)

    #全部特征
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
    regressor = SVR(kernel='linear',C=1,epsilon=0.01)
    kf = KFold(n_splits=kf1, shuffle=True)# 交叉验证
    scores = cross_val_score(regressor, X1, y1, cv=kf, scoring='neg_mean_squared_error')

    # 计算平均 MSE
    average_mse_all = np.mean(-scores)
    ev = 1
    mse_ev = average_mse_1
    EV = len(lis)
    mse_EV = average_mse_all

    def find_greater_than_m(data, m):#寻找data列表中大于m的元素
        # 使用列表推导式找到所有大于m的元素
        result = [x for x in data if x > m]
        return result

    #lis1(名称，分数) lis(名称) lis2(分数)
    lis2 = [item[1] for item in lis1]
    while abs(EV - ev) > 1:
        med = lis2[int((EV + ev)/2)]#EV与ev中值
        new_list = find_greater_than_m(lis2, med)#大于中值的特征
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

        # 创建SVR模型，使用线性核函数
        regressor = SVR(kernel='linear',C=1,epsilon=0.01)
        kf = KFold(n_splits=kf1, shuffle=True)
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
def fscore1_s(pre):#底物预测：计算f-score值
    # 读取Excel文件
    df = pd.read_excel(file_out, engine='openpyxl')
    df = df[~df.isna().any(axis=1)]
    # 提取目标变量y
    y = df[pre]
    # 提取特征X（从第一列到倒数第三列）
    X = df.iloc[:, :-2]

    # 标准化特征X
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

@st.cache_data
def fscore_s(pre,lis1):#底物预测：二分法特征抽提
    lis = [item[0] for item in lis1]
    # 读取Excel文件中的所有sheet
    file_path = file_out
    data = pd.read_excel(file_path)  # 读取
    data = data.dropna(how='any')#删除包含空值的样本点
    # 最高分数特征
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

    # 创建SVR模型，使用线性核函数
    regressor = SVR(kernel='linear',C=1,epsilon=0.01)
    kf = KFold(n_splits=kf1, shuffle=True)
    scores = cross_val_score(regressor, X1, y1, cv=kf, scoring='neg_mean_squared_error')

    # 计算平均 MSE
    average_mse_1 = np.mean(-scores)

    # 全部特征
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

    # 创建SVR模型，使用线性核函数
    regressor = SVR(kernel='linear',C=1,epsilon=0.01)
    kf = KFold(n_splits=kf1, shuffle=True)
    scores = cross_val_score(regressor, X1, y1, cv=kf, scoring='neg_mean_squared_error')

    # 计算平均 MSE
    average_mse_all = np.mean(-scores)
    ev = 1
    mse_ev = average_mse_1
    EV = len(lis)
    mse_EV = average_mse_all

    def find_greater_than_m(data, m):#找出data列表中大于m的元素
        # 使用列表推导式找到所有大于m的元素
        result = [x for x in data if x > m]
        return result

    #lis1(名称，分数) lis(名称) lis2(分数)
    lis2 = [item[1] for item in lis1]
    while abs(EV - ev) > 1:
        med = lis2[int((EV+ev)/2)]#EV与ev间中值
        new_list = find_greater_than_m(lis2, med)#找出所有大于中值的特征
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
        regressor = SVR(kernel='linear',C=1,epsilon=0.01)
        kf = KFold(n_splits=kf1, shuffle=True)
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
def svrm_a(pre,lis):#底物预测SVR模型
    file_path = file_out
    data = pd.read_excel(file_path)  # 读取
    data = data.dropna(how='any')#删除包含空值的样本点
    # SVR
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
    #封装评分器
    r2_scorer = make_scorer(r2_score, greater_is_better=True)
    # 定义SVR模型的超参数搜索空间
    search_space = {
                'C': (1e-2, 1e+2, 'log-uniform'),  # 惩罚项参数 C
                'epsilon': (0.01,1),  # 损失函数的 epsilon
                'kernel': ['linear', 'rbf'],  # 核函数类型
                'gamma': ['scale', 'auto'],  # gamma
            }

    # 使用贝叶斯优化进行超参数搜索
    opt = BayesSearchCV(
                SVR(),  # 模型
                search_space,  # 超参数空间
                n_iter=25,  # 迭代次数
                cv=5,  # 交叉验证次数
                scoring=r2_scorer#评分标准
            )

    # 拟合模型
    opt.fit(X1, y1)

    # 创建SVR模型
    regressor_svr_a = SVR(C=opt.best_params_['C'], epsilon=opt.best_params_['epsilon'],
                                gamma=opt.best_params_['gamma'], kernel=opt.best_params_['kernel'])

    #mse、mae、r2指标
    scores = cross_val_score(regressor_svr_a, X1, y1, cv=kf2, scoring='neg_mean_squared_error')
    scores_mae = cross_val_score(regressor_svr_a, X1, y1, cv=kf2, scoring='neg_mean_absolute_error')
    scores_r2 = cross_val_score(regressor_svr_a, X1, y1, cv=kf2, scoring='r2')
    # 计算平均值
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
    regressor_svr_a.fit(X1, y1)#训练模型
    return regressor_svr_a,mse,scaler_X,scaler_y

@st.cache_data
def svrm_s(pre,lis):#底物预测SVR模型
    file_path = file_out
    data = pd.read_excel(file_path)  # 读取
    data = data.dropna(how='any')#删除包含空值的样本点
    # SVR
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
    r2_scorer = make_scorer(r2_score, greater_is_better=True)#封装评分器
    # 定义SVR模型的超参数搜索空间
    search_space = {
                'C': (1e-2, 1e+2, 'log-uniform'),  # 惩罚项参数 C
                'epsilon': (0.01,1),  # 损失函数的 epsilon
                'kernel': ['linear', 'rbf'],  # 核函数类型
                'gamma': ['scale', 'auto'],  # gamma
            }

    # 使用贝叶斯优化进行超参数搜索
    opt = BayesSearchCV(
                SVR(),  # 模型
                search_space,  # 超参数空间
                n_iter=25,  # 迭代次数
                cv=5,  # 交叉验证次数
                scoring=r2_scorer#评分标准
            )

    # 拟合模型
    opt.fit(X1, y1)

    # 创建SVR模型
    regressor_svr_s = SVR(C=opt.best_params_['C'], epsilon=opt.best_params_['epsilon'],
                                gamma=opt.best_params_['gamma'], kernel=opt.best_params_['kernel'])

    #mse、mae、r2指标
    scores = cross_val_score(regressor_svr_s, X1, y1, cv=kf2, scoring='neg_mean_squared_error')
    scores_mae = cross_val_score(regressor_svr_s, X1, y1, cv=kf2, scoring='neg_mean_absolute_error')
    scores_r2 = cross_val_score(regressor_svr_s, X1, y1, cv=kf2, scoring='r2')
    # 计算平均值
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
    regressor_svr_s.fit(X1,y1)#训练模型
    return regressor_svr_s,mse,scaler_X,scaler_y

@st.cache_data
def plssvrm_a(pre,lis):#产物预测PLS-SVR模型
    file_path = file_out
    data = pd.read_excel(file_path)  # 读取
    data = data.dropna(how='any')#删除包含空值的样本点
    # SVR
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

    # 使用 PLS 进行特征提取，选择提取5个主成分
    pls = PLSRegression(n_components=pls_a_n)

    # 拟合 PLS 模型
    pls.fit(X_train, y_train)

    # 提取特征，得到新的特征矩阵 X_pls（经过降维后的数据）
    X_pls = pls.transform(X_train)
    X_test_pls = pls.transform(X_test)

    X1 = X_pls
    y1 = y_train
    r2_scorer = make_scorer(r2_score, greater_is_better=True)#封装评分器
    # 定义SVR模型的超参数搜索空间
    search_space = {
                'C': (1e-2, 1e+2, 'log-uniform'),  # 惩罚项参数 C
                'epsilon': (0.01,1),  # 损失函数的 epsilon
                'kernel': ['linear', 'rbf'],  # 核函数类型
                'gamma': ['scale', 'auto'],  # gamma
            }

    # 使用贝叶斯优化进行超参数搜索
    opt = BayesSearchCV(
                SVR(),  # 模型
                search_space,  # 超参数空间
                n_iter=25,  # 迭代次数
                cv=5,  # 交叉验证次数
                scoring=r2_scorer#评分标准
            )

    # 拟合模型
    opt.fit(X1, y1)

    # 创建SVR模型
    regressor_plssvr_a = SVR(C=opt.best_params_['C'], epsilon=opt.best_params_['epsilon'],
                                gamma=opt.best_params_['gamma'], kernel=opt.best_params_['kernel'])
    #mse、mae、r2指标
    scores = cross_val_score(regressor_plssvr_a, X1, y1, cv=kf2, scoring='neg_mean_squared_error')
    scores_mae = cross_val_score(regressor_plssvr_a, X1, y1, cv=kf2, scoring='neg_mean_absolute_error')
    scores_r2 = cross_val_score(regressor_plssvr_a, X1, y1, cv=kf2, scoring='r2')
    # 计算平均值
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
    regressor_plssvr_a.fit(X1, y1)#训练模型
    return regressor_plssvr_a,mse,scaler_X,scaler_y,pls.transform

@st.cache_data
def plssvrm_s(pre,lis):#底物预测PLS-SVR模型
    file_path = file_out
    data = pd.read_excel(file_path)  # 读取
    data = data.dropna(how='any')#删除包含空值的样本点
    # SVR
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

    # 使用 PLS 进行特征提取，选择提取7个主成分
    pls = PLSRegression(n_components=pls_s_n)

    # 拟合 PLS 模型
    pls.fit(X_train, y_train)

    # 提取特征，得到新的特征矩阵 X_pls（经过降维后的数据）
    X_pls = pls.transform(X_train)
    X_test_pls = pls.transform(X_test)

    X1 = X_pls
    y1 = y_train
    r2_scorer = make_scorer(r2_score, greater_is_better=True)#封装评分器
    # 定义SVR模型的超参数搜索空间
    search_space = {
                'C': (1e-2, 1e+2, 'log-uniform'),  # 惩罚项参数 C
                'epsilon': (0.01,1),  # 损失函数的 epsilon
                'kernel': ['linear', 'rbf'],  # 核函数类型
                'gamma': ['scale', 'auto'],  #gamma
            }

            # 使用贝叶斯优化进行超参数搜索
    opt = BayesSearchCV(
                SVR(),  # 模型
                search_space,  # 超参数空间
                n_iter=25,  # 迭代次数
                cv=5,  # 交叉验证次数
                scoring=r2_scorer#评分标准
            )

    # 拟合模型
    opt.fit(X1, y1)

    # 创建SVR模型，使用RBF核函数
    regressor_plssvr_s = SVR(C=opt.best_params_['C'], epsilon=opt.best_params_['epsilon'],
                                gamma=opt.best_params_['gamma'], kernel=opt.best_params_['kernel'])
    # mse、mae、r2指标
    scores = cross_val_score(regressor_plssvr_s, X1, y1, cv=kf2, scoring='neg_mean_squared_error')
    scores_mae = cross_val_score(regressor_plssvr_s, X1, y1, cv=kf2, scoring='neg_mean_absolute_error')
    scores_r2 = cross_val_score(regressor_plssvr_s, X1, y1, cv=kf2, scoring='r2')
    # 计算平均值
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
    regressor_plssvr_s.fit(X1, y1)#训练模型
    return regressor_plssvr_s,mse,scaler_X,scaler_y,pls.transform


#预测
@st.cache_data
def xd00():#添加发酵时间
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

@st.cache_data
def xd11():#添加变化特征
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
        columns_to_process = df.columns[1:m1 - 1]
        # 计算差值并添加到DataFrame中
        for col in columns_to_process:
            df[f'{col}_变化'] = df[col].diff()
        # 将处理后的DataFrame存储在字典中
        processed_sheets[sheet_name] = df
    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

@st.cache_data
def xd22():#添加变化特征
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
        for col in range(m1, n3):
            df[f'变化率_{df.columns[col]}'] = df.iloc[:, col] / df['发酵时间']
        processed_sheets[sheet_name] = df
    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

@st.cache_data
def xd33():#添加变化率特征
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
        for col in range(1, n4):
            # 新列的值为上一行的数据
            df[f'上一_{df.columns[col]}'] = df.iloc[:, col].shift(1)
        processed_sheets[sheet_name] = df
    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

@st.cache_data
def xd44():#添加多项式特征
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
        for col in range(1, n5):
            new_col_name = f"{df.columns[col]}_squared"  # 新特征列名称
            df[new_col_name] = df.iloc[:, col] ** 2  # 计算平方并添加新列
        processed_sheets[sheet_name] = df
    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

@st.cache_data
def xd55():#添加累积特征
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
    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

@st.cache_data
def xd66():#添加转化率特征
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
            df['菌体mL转化率'] = df['菌浓mL/50mL_变化'] / df[f'{df.columns[1]}_变化']

        processed_sheets[sheet_name] = df
    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

@st.cache_data
def xd77():#添加生物学特征
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
    with pd.ExcelWriter(file_out_in) as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

st.title("基于SVR的生物过程预测")
# 创建一个按钮
if st.button('说明书'):
    with st.expander("基于SVR的生物过程预测使用说明书"):
        st.markdown("1. 系统概述")
        st.markdown("该网站主要分为两个部分：生物过程预测，模型预测控制（可选）")
        st.markdown("2. 生物过程预测")
        st.markdown("生物过程预测包含5个模块")
        st.markdown("2.1 文件读取")
        st.markdown("功能：读取选择的包含训练数据的Excel文件。")
        st.markdown("注意：文件应包含多个sheet（每个sheet表示一个批次），每个sheet的格式如下：横轴为测量点，纵轴为特征名称，特征名称顺序为：发酵周期/h、底物、产物、其他特征1、其他特征2...（前三个特征不可改变，菌浓相关特征应命名为“菌浓mL/50mL”或“菌浓g/50mL”）。测量时间间隔应尽可能相等。")
        st.markdown("操作：点击“文件读取”按钮以读取训练数据，随后可点击“文件预览”按钮以浏览读取的训练数据。在多选框中选择操作参数")
        st.markdown("2.2 特征拓展")
        st.markdown("功能：对已有的特征进行拓展，丰富数据特征，更好的学习其中的规律")
        st.markdown("建议：建议全选所有特征进行拓展。")
        st.markdown("操作：多选框选择拓展类型，点击“开始拓展”按钮进行特征拓展，拓展完成后，数据将保存为名为“d3”的Excel文件。点击“处理数据预览”按钮以预览拓展后的数据。")
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
        st.markdown("②输入发酵周期")
        st.markdown("③点击“输入数据预处理”按钮，数据将根据“特征拓展”中的设置进行相同类型的拓展。")
        st.markdown("④预处理完成后，展示处理后的待预测数据。")
        st.markdown("⑤选择“开始预测产物”或“开始预测底物”按钮进行预测。选择一个训练好的模型进行预测（推荐选择MSE较低的模型）。")
        st.markdown("3. 模型预测控制（可选）")
        st.markdown("操作：")
        st.markdown("①在复选框中选择需要进行优化的控制变量。")
        st.markdown("②点击“输入上下限按钮”，设置各控制变量的下限与上限。控制变量的取值需符合实际情况。")
        st.markdown("③点击“开始”按钮，开始进行模型预测控制。")
        st.markdown("4. 缓存清理")
        st.markdown("若需要清理缓存，请点击网页最下方的“清理缓存”按钮。")
        st.markdown("5. 注意事项")
        st.markdown("在进行模型训练、特征抽提等需要较长时间的操作时，请耐心等待，避免频繁操作导致系统不稳定")
        st.markdown("本系统仅适用于符合规定格式的训练数据，上传的数据应严格按照说明书中要求的格式进行整理")
        st.markdown("6. 版权说明")
        st.markdown("孙展鵾 李友元(yyli@ecust.edu.cn) * 华东理工大学生物工程学院 Copyright 2025")

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

    mpc_lis = df1.columns#原始特征
    #操作参数
    options_mpc_0 = st.multiselect(
        label='请选择操作参数',
        options=mpc_lis,
        default=None,
        format_func=str,
    )

st.header("2.特征拓展")
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
    # 添加发酵时间
    n1 = xd0()
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
    xls = pd.ExcelFile(file_out)
    df2 = pd.read_excel(xls)
    if st.button('处理数据预览'):
        st.session_state.aa = 'aa'
    # 显示结果
    if 'aa' in st.session_state:
        st.subheader('数据预览')
        st.write(df2.head(20))

st.header("3.特征抽提")
if st.button('开始特征抽提'):
    st.session_state.c = 'c'
if 'c' in st.session_state:
    #产物
    st.write('预测产物特征抽提:')
    aflist = fscore1_a(pre1)#计算f-score值
    aEV = fscore_a(pre1, aflist)#特征抽提：最佳特征数
    acid_con = list(dict.fromkeys([item[0] for item in aflist][:aEV] + options_mpc_0))#操作参数结合抽提特征
    st.write('选择特征数：', len(acid_con))
    st.write(acid_con)
    #底物
    st.write('预测底物特征抽提：')
    sflist = fscore1_s(pre2)#计算f-score值
    sEV = fscore_s(pre2, sflist)#特征抽提：最佳特征数
    sur_con = list(dict.fromkeys([item[0] for item in sflist][:sEV] + options_mpc_0))#操作参数结合抽提特征
    st.write('选择特征数：', len(sur_con))
    st.write(sur_con)

st.header("4.模型训练")
if st.button('开始训练产物预测模型'):
    st.session_state.d = 'd'
if 'd' in st.session_state:#产物预测模型
    svrmse_aa = '未预测'
    plssvrmse_aa = '未预测'
    svrm_aa, svrmse_aa,sacX,sacy = svrm_a(pre1, acid_con)#SVR模型预训练与优化
    plssvrm_aa, plssvrmse_aa,psacX,psacy,ps_aa = plssvrm_a(pre1, acid_con)#PLS-SVR模型预训练与优化
    svrmse_aa=round(svrmse_aa,4)
    plssvrmse_aa=round(plssvrmse_aa,4)

if st.button('开始训练底物预测模型'):
    st.session_state.e = 'e'
if 'e' in st.session_state:#底物预测模型
    svrmse_ss = '未预测'
    plssvrmse_ss = '未预测'
    svrm_ss, svrmse_ss ,sscX,sscy= svrm_s(pre2, sur_con)#SVR模型预训练与优化
    plssvrm_ss, plssvrmse_ss ,psscX,psscy,ps_ss= plssvrm_s(pre2, sur_con)#PLS-SVR模型预训练与优化
    svrmse_ss = round(svrmse_ss, 4)
    plssvrmse_ss = round(plssvrmse_ss, 4)

st.header("5.预测")
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
        m1 = xd00()#发酵时间
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
        prdata = df22.iloc[-1]#提取最后一行（待预测数据）
        st.write(prdata)

if st.button('开始预测产物'):
    st.session_state.g = 'g'
if 'g' in st.session_state:
    #选择模型
    model = st.radio(
            label='请选择产物预测模型',
            options=(f'SVR   MSE:{svrmse_aa}', f'PLS-SVR   MSE:{plssvrmse_aa}'),
            index=1,
            format_func=str,
            help='推荐使用MSE较小的模型'
        )
    if model == f'SVR   MSE:{svrmse_aa}':#选择SVR模型
        features = acid_con
        X = df22[features].to_numpy()
        predata1 = sacX.transform(X)#归一化
        y_preddata1_a = svrm_aa.predict(predata1[-1].reshape(1,-1))#预测
        data_original = sacy.inverse_transform(y_preddata1_a.reshape(1,-1))#复原数据
        st.write(f'底物消耗速率为{(data_original[0]-prdata.iloc[1])/prdata.iloc[m1-1]}')
        st.write(f'{perld}后{df22.columns[2]}预计浓度为{data_original[0]}')

    elif model ==f'PLS-SVR   MSE:{plssvrmse_aa}':#选择PLS-SVR模型
        features = acid_con
        X = df22[features].to_numpy()
        predata1 = psacX.transform(X)#归一化
        predata1 = ps_aa(predata1[-1].reshape(1,-1))#pls降维
        y_preddata1_a = plssvrm_aa.predict(predata1)#预测
        data_original = psacy.inverse_transform(y_preddata1_a.reshape(1,-1))#复原数据
        st.write(f'底物消耗速率为{(data_original[0]-prdata.iloc[1])/prdata.iloc[m1-1]}')
        st.write(f'{perld}后{df22.columns[2]}预计浓度为{data_original[0]}')

if st.button('开始预测底物'):
    st.session_state.h = 'h'
if 'h' in st.session_state:
    #选择模型
    model = st.radio(
            label='请选择底物预测模型',
            options=(f'SVR   MSE:{svrmse_ss}', f'PLS-SVR   MSE:{plssvrmse_ss}'),
            index=1,
            format_func=str,
            help='推荐使用MSE较小的模型'
        )
    if model == f'SVR   MSE:{svrmse_ss}':#选择SVR模型
        features = sur_con
        X = df22[features].to_numpy()
        predata2 = sscX.transform(X)#归一化
        y_preddata2_s = svrm_ss.predict(predata2[-1].reshape(1,-1))#预测
        data_original_s = sscy.inverse_transform(y_preddata2_s.reshape(1,-1))#复原数据
        st.write(f'底物消耗速率为{(data_original_s[0]-prdata.iloc[1])/prdata.iloc[m1-1]}')
        st.write(f'{perld}后{df22.columns[1]}预计浓度为{data_original_s[0]}')

    elif model ==f'PLS-SVR   MSE:{plssvrmse_ss}':#选择PLS-SVR模型
        features = sur_con
        X = df22[features].to_numpy()
        predata2 = psscX.transform(X)#归一化
        predata2 = ps_ss(predata2[-1].reshape(1,-1))#预测
        y_preddata2_s = plssvrm_ss.predict(predata2)#pls降维
        data_original_s = psscy.inverse_transform(y_preddata2_s.reshape(1,-1))#预测
        st.write(f'底物消耗速率为{(data_original_s[0]-prdata.iloc[1])/prdata.iloc[m1-1]}')#复原数据
        st.write(f'{perld}后{df22.columns[1]}预计浓度为{data_original_s[0]}')

st.header("6.模型预测控制")
#需要优化的操作参数
options_mpc = st.multiselect(
    label='请选择需要优化的操作参数',
    options=options_mpc_0,
    default=None,
    format_func=str,
)

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
    state_params = mpc_x.to_numpy()#状态
    initial_control_params = mpc_x[options_mpc].to_numpy()#初始操作参数
    indices = [acid_con.index(x) for x in options_mpc]#获得操作参数索引
    def objective(control_params):
        # 将改变的操作参数与抽提特征组合
        for i in indices:
            state_params[i] = control_params[indices.index(i)]
        input_features = sacX.transform(state_params.reshape(1, -1))#归一化
        # 使用SVR模型进行预测
        pre_mpc = svrm_aa.predict(input_features)#预测
        prediction = sacy.inverse_transform(pre_mpc.reshape(1, 1))#复原
        # 目标是最大化预测值，因此返回负值（minimize）
        return -prediction[0]

    # 调用优化函数
    result = minimize(objective, initial_control_params, bounds=bounds, method="Nelder-Mead",options={'maxiter': 1000, 'disp': True})#Nelder-Mead方法适用于不平滑函数优化
    # 输出结果
    optimal_control_params = result.x
    max_prediction = -result.fun  # 因为返回的是负值，反转回来

    for i in range(len(options_mpc)):
        st.write(f"{options_mpc[i]}最优控制参数: {optimal_control_params[i]}")
    st.write(f"最大预测值: {max_prediction}")

# 清除缓存
st.header("7.缓存清理")
if st.button('清理缓存'):
    st.cache_data.clear()
    st.write("缓存已清理！")

st.markdown("版权说明： 孙展鵾 李友元(yyli@ecust.edu.cn) * 华东理工大学生物工程学院 Copyright 2025")#作者 版权说明





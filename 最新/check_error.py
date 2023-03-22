import numpy as np
# 定义孤立森林模型的树数量和子采样大小
n_trees = 100
subsample_size = 256

# 创建字典以存储每个特征的孤立森林模型
clfs = {}
train_data = pd.DataFrame(train_data)
# 循环遍历数据集中的每个特征并训练一个孤立森林模型
for column_name in train_data.columns:
    # 提取特征值
    column_data = train_data[column_name].values.reshape(-1, 1)
    
    # 训练孤立森林模型
    model = IsolationForest(n_estimators=n_trees, max_samples=subsample_size, random_state=42)
    model.fit(column_data)



    # 存储已训练的模型
    clfs[column_name] = model
outliers = queue.Queue(5)
outliers1 = queue.Queue(5)
def detect_outliers(predict_data, original_data,outliers):
    # 判断PM2.5、PM10和NO2是否超过正常范围
    if predict_data[0] > 500 or predict_data[0] < 0:
        outliers.append('PM2.5')
    if predict_data[1] > 600 or predict_data[1] < 0:
        outliers.append('PM10')
    if predict_data[2] > 200 or predict_data[2] < 0:
        outliers.append('NO2')
    
    # 判断温度和湿度是否超过正常范围
    if predict_data[3] > 50 or predict_data[3] < -20:
        outliers.append('temperature')
    if predict_data[4] > 100 or predict_data[4] < 0:
        outliers.append('humidity')
    
    # 判断预测值与原始值之间是否存在明显差异
    diff = np.abs(predict_data - original_data)
    if np.max(diff) > 20:
        outliers.append('diff_max')
    
    # 判断数据在短时间内是否出现剧烈波动
    if len(outliers) == 0:
        std = np.std(original_data)
        mean = np.mean(original_data)
        z_score = np.abs((predict_data - mean) / std)
        if np.max(z_score) > 3:
            outliers.append('z_score_max')

    return outliers
# def check_error(predict_data):
#     for i in predict_data:
#         if clfs[i].predict(predict_data[i]) == -1:
#             return True
# def detect_outliers2(predict_data, original_data,outliers,outliers1):
#        if check_error(predict_data[1:7]):
#            outliers.append((predict_data,original_data)) 
#            for i,value in enumerate(predict_data[1:7]):
#                if clfs[i].predict(predict_data[i]) == -1:
#                    outliers1.append(value)
#        if outliers.full():


           

def diff(predict_data,original_data):
    # 定义函数来计算预测数据和原始数据之间的差异
    diff = 0
    for i,value in enumerate(predict_data):
        diff+=abs(value-original_data[i]);
    return diff/len(predict_data);
# 这个函数对多个指标进行了异常检测。如果有任何异常，则将其添加到列表中返回。接下来，我们需要修改check_data函数以使用新的异常检测函数，并更精确地分类异常数据。

def check_data(predict_data, original_data, threshold, threshold2, row, queue):
    outliers = detect_outliers(predict_data, original_data)
    if len(outliers) == 0:
        # 如果没有异常，则将其视为可校准数据
        data1[row] = predict_data
        queue.append(row)
        total = sum(queue)
        if len(queue) == queue.maxsize and total / len(queue) > threshold2:
            # 如果队列已满，并且平均误差小于阈值，则重新训练模型
            train_size = int(row)
            train_data = data_scaled[:train_size, :]
            test_data = data_scaled[train_size:, :]
            X_train, Y_train = create_dataset(train_data, look_back)
            X_test, Y_test = create_dataset(test_data, look_back)
            model.fit(X_train, Y_train, epochs=400, batch_size=40, validation_data=(X_test, Y_test), verbose=1)
            queue.clear()
        return queue, 'calibration'
    else:
        # 如果有异常，则判断异常类型
        if 'z_score_max' in outliers or 'diff_max' in outliers:
            print("微站损坏：", row)
            return queue, 'damage'
        else:
            print("微站异常：", row, outliers)
            return queue, 'abnormal'

# 在新版本的check_data函数中，我们首先使用detect_outliers函数检测数据是否异常。如果没有任何异常，它就是可校准数据；否则，我们会进一步判断异常类型。如果数据在短时间内出现剧烈波动或者存在明显的偏离正常范围的值，则被认为是微站损坏；否则被认为是微站异常。

# 最后，我们需要将修改后的check_data函数应用于主循环中。以下是新的主循环：

train_len = int(len(data) * 0.3)
import queue
q = queue.Queue(maxsize=3)
for i in range(train_len,len(data)):
    predict_data = model.predict(data[i][0])
    #反归一化
    predict_data = scaler.inverse_transform(predict_data)
    original_data = data[i]
    q, status = check_data(predict_data[0],original_data[1:7],0.05,0.3,i,q)
    if status == 'damage':
    # 如果是微站损坏，则记录日志并发送警报
        print("微站损坏：", i)
    # TODO: 发送警报到管理系统
    elif status == 'abnormal':
    # 如果是微站异常，则记录日志和异常类型，并进行维护
        print("微站异常：", i)
        print("异常类型：", q[-1])
    # TODO: 记录日志和异常类型，进行维护操作

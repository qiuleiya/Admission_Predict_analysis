import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error
df = pd.read_csv("Admission_Predict_Ver1.1.csv",sep = ",")

#第一列序号与入学概率之间没有相关性，所以删除掉
# it may be needed in the future.
serialNo = df["Serial No."].values
df.drop(["Serial No."],axis=1,inplace = True)
df.drop(["GRE Score","TOEFL Score"],axis=1,inplace = True)
# 入学机会为预测字段
y = df["Chance of Admit "].values
x = df.drop(["Chance of Admit "],axis=1)
# 准备训练数据
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)
#数据归一化
from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))
x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])

#线性回归

lr = LinearRegression()
lr.fit(x_train,y_train)
print("线性回归预测:序号1真实入学概率: " + str(y_test[1]) + " -> 预测概率: " + str(lr.predict(x_test.iloc[[1],:])))
print("线性回归预测:序号50真实入学概率: " + str(y_test[50]) + " -> 预测概率: " + str(lr.predict(x_test.iloc[[50],:])))

y_head_lr = lr.predict(x_test)
y_head_lr_train = lr.predict(x_train)
print("线性回归预测模型测试数据的拟合优度：", r2_score(y_test,y_head_lr))
print("线性回归预测模型训练数据的拟合优度： ", r2_score(y_train,y_head_lr_train))
n1=np.sqrt(np.mean((y_test - y_head_lr) ** 2))
print("线性回归预测模型测试数据的均方误差根为:", n1)
c = [i for i in range(1,101,1)]
plt.plot(c,y_test, color = 'green', linewidth = 2, label='Real')
plt.plot(c,y_head_lr, color = 'red', linewidth = 2, label='Predicted')
plt.grid(alpha = 0.3)
plt.legend()
plt.title('LinearRegression Real vs Predicted')
plt.show()

#决策树
dtr = DecisionTreeRegressor(max_depth=3,min_samples_leaf=40)
#dtr = DecisionTreeRegressor(max_depth=5)
dtr.fit(x_train,y_train)
print("决策树回归预测:序号1真实入学概率: : " + str(y_test[1]) + " -> 预测值: " + str(dtr.predict(x_test.iloc[[1],:])))
print("决策树回归预测:序号50真实入学概率: : " + str(y_test[50]) + " -> 预测值: " + str(dtr.predict(x_test.iloc[[50],:])))

y_head_dtr = dtr.predict(x_test)
y_head_dtr_train = dtr.predict(x_train)
print("决策树回归模型测试数据拟合优度: ", r2_score(y_test,y_head_dtr))
print("决策树回归模型训练数据拟合优度: ", r2_score(y_train,y_head_dtr_train))
n2=np.sqrt(np.mean((y_test - y_head_dtr) ** 2))
print("决策树回归预测模型测试数据的均方误差根为:", n2)

c = [i for i in range(1,101,1)]
plt.plot(c,y_test, color = 'green', linewidth = 2, label='Real')
plt.plot(c,y_head_dtr, color = 'red', linewidth = 2, label='Predicted')
plt.grid(alpha = 0.3)
plt.legend()
plt.title('DecisionTreeRegressor Real vs Predicted')
plt.show()

#随机森林
rfr = RandomForestRegressor(n_estimators = 80, min_samples_leaf=15,random_state =42,max_depth=3)
rfr.fit(x_train,y_train)
y_head_rfr = rfr.predict(x_test)
print("随机森林回归预测:序号1真实入学概率: " + str(y_test[1]) + " -> 预测值: " + str(rfr.predict(x_test.iloc[[1],:])))
print("随机森林回归预测:序号50真实入学概率:  " + str(y_test[50]) + " -> 预测值: " + str(rfr.predict(x_test.iloc[[50],:])))
y_head_rf_train = rfr.predict(x_train)
print("随机森林回归模型测试数据拟合优度: ", r2_score(y_test,y_head_rfr))
print("随机森林回归模型训练数据拟合优度: ", r2_score(y_train,y_head_rf_train))
n3=np.sqrt(np.mean((y_test - y_head_rfr) ** 2))
print("随机森林回归预测模型测试数据的均方误差根为:",n3 )


c = [i for i in range(1,101,1)]
plt.plot(c,y_test, color = 'green', linewidth = 2, label='Real')
plt.plot(c,y_head_rfr, color = 'red', linewidth = 2, label='Predicted')
plt.grid(alpha = 0.3)
plt.legend()
plt.title('RandomForestRegressor Real vs Predicted')
plt.show()

#回归模型预测值对比
red = plt.scatter(np.arange(0,100,5),y_head_lr[0:100:5],color = "red")
green = plt.scatter(np.arange(0,100,5),y_head_rfr[0:100:5],color = "green")
blue = plt.scatter(np.arange(0,100,5),y_head_dtr[0:100:5],color = "blue")
black = plt.scatter(np.arange(0,100,5),y_test[0:100:5],color = "black")
plt.title("Comparison of Regression Algorithms")
plt.xlabel("Index of Candidate")
plt.ylabel("Chance of Admit")
plt.legend((red,green,blue,black),('LR', 'RFR', 'DTR', 'REAL'))
plt.show()

#回归模型拟合优度对比
y = np.array([r2_score(y_test,y_head_lr),r2_score(y_test,y_head_dtr),r2_score(y_test,y_head_rfr)])
x = ["LinearRegression","DecisionTreeReg.","RandomForestReg."]
sns.barplot(x,y)
plt.title("Regression Algorithms")
plt.xlabel("Regressor")
plt.ylabel("r2_score")
plt.show()

#误差
y = np.array([n1,n2,n3])
x = ["LinearRegression","DecisionTreeReg.","RandomForestReg."]
sns.barplot(x,y)
plt.title("Regression Algorithms")
plt.xlabel("Regressor")
plt.ylabel("r2_score")
plt.show()

print("线性回归预测模型测试数据的拟合优度：", r2_score(y_test,y_head_lr))
print("决策树回归模型测试数据拟合优度: ", r2_score(y_test,y_head_dtr))
print("随机森林回归模型测试数据拟合优度: ", r2_score(y_test,y_head_rfr))

print("线性回归预测模型测试数据的均方误差根为:", n1)
print("决策树回归预测模型测试数据的均方误差根为:", n2)
print("随机森林回归预测模型测试数据的均方误差根为:",n3 )
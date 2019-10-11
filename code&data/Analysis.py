import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


df = pd.read_csv("Admission_Predict_Ver1.1.csv",sep = ",")
print("该数据集总共有",len(df.columns),"列:")
for x in df.columns:
    sys.stdout.write(str(x)+", ")

print("数据集的样本和特征数量如下\n")
print(df.info())
print("前五行数据如下：\n",df.head())
print("末五行数据如下：\n",df.tail())
print(df.describe())
fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.title('A_P.corr heatmap')
plt.show()

d1=df['GRE Score']
print("GRE成绩：\n",d1.describe())
d2=df['TOEFL Score']
print("托福成绩：\n",d2.describe())
d6=df['CGPA']
print("CGPA：\n",d6.describe())
d3=df['University Rating']
d4=df['SOP']
d5=df['LOR ']
d7=df['Research']
d8=df['Chance of Admit ']
x = df.drop(["Serial No."],axis=1)
#多变量图
sns.pairplot(x)
#各项数据分析统计
#GRE成绩

sns.distplot(d1)
plt.title('Distributed GRE Scores of Applicants')
plt.show()
#托福成绩

sns.distplot(d2)
plt.title('Distributed TOEFL Scores of Applicants')
plt.show()
#大学等级
sns.countplot(d3)
plt.title('University Rating')
plt.ylabel('Count')
plt.show()
#SOP
sns.countplot(d4)
plt.title('SOP')
plt.ylabel('Count')
plt.show()
#LOR
sns.countplot(d5)
plt.title('LOR ')
plt.ylabel('Count')
plt.show()

#CGCP成绩
sns.distplot(d6)
plt.title('CGPA Distribution of Applicants')
plt.show()
#科研经历
GRE = pd.DataFrame(df['GRE Score'])
RES_Count = df.groupby(['Research']).count()
RES_Count = RES_Count['GRE Score']
RES_Count = pd.DataFrame(RES_Count)
RES_Count.rename({'GRE Score': 'Count'}, axis=1, inplace=True)
RES_Count.rename({0: 'No Research', 1:'Research'}, axis=0, inplace=True)
plt.pie(x=RES_Count['Count'], labels=RES_Count.index, autopct='%1.1f%%')
plt.title('Research', pad=5, size=30)
plt.show()

#GRE、toefl、CGPA和chance关系

sns.regplot(d1,d8,color='g')
plt.title("GRE Score")
plt.xlabel("GRE Score")
plt.ylabel("Chance of Admit ")
plt.show()

sns.regplot(d2,d8,color='g')
plt.title("TOEFL Score ")
plt.xlabel("TOEFL Score")
plt.ylabel("Chance of Admit ")
plt.show()


sns.regplot(d6,d8,color='g')
plt.title("CGPA VS chance")
plt.xlabel("CGPA")
plt.ylabel("Chance of Admit ")
plt.show()


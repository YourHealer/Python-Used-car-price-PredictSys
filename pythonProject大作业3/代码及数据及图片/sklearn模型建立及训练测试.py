import numpy as np              #引入numpy
import pandas as pd             #引入pandas

#数据可视化
import matplotlib.pyplot as plt #引入matplotlib
import seaborn as sns           #引入seaborn
import missingno as msno        #引入missingno

#机器学习
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,LassoCV
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestRegressor
seed=1234

#读取数据
data=pd.read_csv("cleanData.csv")

#分析无关数据清理
data=data.drop(['order','region_url','lat','long','posting_date','url','VIN','id','image_url','paint_color'],axis=1)
# print(data)

#缺失值missingno图判断
sns.set(style="ticks")
msno.matrix(data)
# plt.show()

#cylinders参数缺失值及其余非数值的均值修正
data['cylinders'].fillna('6', inplace=True)
for i in range(len(data)):
    if(data['cylinders'][i]=='other'):
        data.loc[i,'cylinders'] = '6'
#cylinders参数数值的取出
data['cylinders'] = data['cylinders'].apply(lambda x: x.split(' ')[0])
#cylinders参数数据类型转变
data_types_dict = {'cylinders': 'int64'}
data = data.astype(data_types_dict)
# print(data['cylinders'])

#condition参数数据类型赋值转变
for i in range(len(data)):
    if(data['condition'][i]=='new'):
        data.loc[i,'condition'] = 100
    elif (data['condition'][i] == 'like new'):
        data.loc[i,'condition'] = 90
    elif (data['condition'][i] == 'excellent'):
        data.loc[i,'condition'] = 80
    elif (data['condition'][i] == 'good'):
        data.loc[i,'condition'] = 70
    elif (data['condition'][i] == 'fair'):
        data.loc[i,'condition'] = 60
    elif (data['condition'][i] == 'salvage'):
        data.loc[i,'condition'] = 50
data_types_dict = {'condition': 'int64'}
data = data.astype(data_types_dict)
# print(data['condition'])

#title_status参数数据类型赋值转变
for i in range(len(data)):
    if (data['title_status'][i] == 'clean'):
        data.loc[i, 'title_status'] = 100
    else:
        data.loc[i, 'title_status'] = 80
data_types_dict = {'title_status': 'int64'}
data = data.astype(data_types_dict)

#特征相关性
cormatrix = data.corr()
cormatrix *= np.tri(*cormatrix.values.shape,k=-1).T
cormatrix = cormatrix.stack()
cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()
cormatrix.columns = ["FirstVariable","SecondVarible","Correlation"]
# print(cormatrix)

corr_all=data.corr()
mask=np.zeros_like(corr_all,dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#绘制热力图
f,ax=plt.subplots(figsize=(1,9))
sns.heatmap(corr_all,mask=mask,square=True,linewidths=.5,ax=ax,cmap="BuPu")
# plt.show()

#绘制不同参数下的回归图
sns.lmplot('price','odometer',data,hue='manufacturer',col='manufacturer',row='fuel',palette='plasma',fit_reg=True)
sns.lmplot('price','odometer',data,hue='drive',col='drive',row='fuel',palette='plasma',fit_reg=True)
# plt.show()

#进行数据标准化
target=data.price
regressors=[x for x in data.columns if x not in ['price']]
features=data.loc[:,regressors]

num=['year','condition','cylinders','odometer','title_status']
standard_scaler=StandardScaler()
features[num]=standard_scaler.fit_transform(features[num])
# print(features.head())

classes=['region','manufacturer','model','fuel','transmission','drive','type','state']
dummies=pd.get_dummies(features[classes])
features=features.join(dummies).drop(classes,axis=1)

# print("In total:",features.shape)
# print(features.head())

#划分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=seed)

# lasso回归
# 指定alphas的范围
alphas = 2. ** np.arange(2, 12)
scores = np.empty_like(alphas)

for i, a in enumerate(alphas):
    lasso = Lasso(random_state=seed)
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    scores[i] = lasso.score(X_test, y_test)

# 交叉验证cross validation
lassocv = LassoCV(cv=10, random_state=seed)
lassocv.fit(features, target)
lassocv_score = lassocv.score(features, target)
lassocv_alpha = lassocv.alpha_

#绘制折线图
plt.figure(figsize=(10, 4))
plt.plot(alphas, scores, '-ko')
plt.axhline(lassocv_score, color='red')
plt.xlabel(r'$\alpha$')
plt.ylabel('CV Score')
plt.xscale('log', basex=2)
sns.despine(offset=15)
plt.savefig("line.jpg")
print('CV results:', lassocv_score, lassocv_alpha)

# lassocv coefficients
coefs = pd.Series(lassocv.coef_, index = features.columns)

#绘制交叉验证条形图
coefs = pd.concat([coefs.sort_values().head(5), coefs.sort_values().tail(5)])# 展示前5个和后5个
plt.figure(figsize = (10, 4))
coefs.plot(kind = "barh", color = 'salmon')
plt.title("Coefficients in the Lasso Model")
# plt.show()

# 将上面计算出来的Alphas 代入
model_l1 = LassoCV(alphas = alphas, cv = 10, random_state = seed).fit(X_train, y_train)
y_pred_l1 = model_l1.predict(X_test)

model_l1.score(X_test, y_test)

#绘制残差图
plt.rcParams['figure.figsize'] = (6.0, 6.0)
preds = pd.DataFrame({"preds": model_l1.predict(X_train), "true": y_train})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals", kind = "scatter", color = 'salmon')
plt.show()

# 计算偏差指标：MSE和R2
mse = mean_squared_error(y_test, y_pred_l1)
print('MSE: %2.3f' % mse)
r2 = r2_score(y_test, y_pred_l1)
print('R2: %2.3f' % r2)

#结果预测
d = {'true' : list(y_test),'predicted' : pd.Series(y_pred_l1)}
uu=pd.DataFrame(d)
cnt=0
print(uu)
for i in range(len(uu)):
    if(0.7*uu['true'][i]<uu['predicted'][i]<1.3*uu['true'][i]):
        cnt+=1
# print(uu.dtypes)
print("准确率：",cnt/len(uu))
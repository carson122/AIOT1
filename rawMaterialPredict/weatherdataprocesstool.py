
import numpy as np
import pandas as pd 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

#天氣資料欄位值處理
def dataMean(s1,colName,groupbyIndex=list):
    if s1[colName].max()>1:
        dataMean=s1[s1[colName]<s1[colName].max()]#拿掉沒資料的
    else:
        dataMean=s1
    dataMean=dataMean.groupby(groupbyIndex)[colName].mean()
    return dataMean

def dataSum(s1,colName,groupbyIndex):
    if s1[colName].max()>1:
        dataSum=s1[s1[colName]<s1[colName].max()]#拿掉沒資料的
    else:
        dataSum=s1
    dataSum=dataSum.groupby(groupbyIndex)[colName].sum()
    return dataSum

def dataMax(s1,colName,groupbyIndex):
    if s1[colName].max()>1:
        dataMax=s1[s1[colName]<s1[colName].max()]#拿掉沒資料的
    else:
        dataMax=s1
    dataMax=dataMax.groupby(groupbyIndex)[colName].max()
    return dataMax

def dataMin(s1,colName,groupbyIndex):
    if s1[colName].max()>1:
        dataMin=s1[s1[colName]<s1[colName].max()]#拿掉沒資料的
    else:
        dataMax=s1
    dataMin=dataMin.groupby(groupbyIndex)[colName].min()
    return dataMin

#價格報酬率label建立
def priceReturnFeature(stock_id,route):
    price=pd.read_pickle('/Users/benbilly3/Desktop/資策會專題/rawMaterialPricePrediction/RM_Price/rawMaterialPrice.pickle')
    price=price.loc[stock_id]
    price['next']=price['Close'].shift(-route)
    price['return']=price['next']/price['Close']
    price=price.dropna()
    return price


#機器學習視覺化

def plot_decision_regions(X, y, classifier, resolution=0.02):# classifier為選取器選擇

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('palegreen','pink', 'palegreen', 'lightskyblue', 'snow', 'lemonchiffon')
    colors2 = ('forestgreen','indianred', 'royalblue', 'gray', 'darkgoldenrod')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    cmap2 = ListedColormap(colors2[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    color=cmap2(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import math
from libtiff import TIFF
import seaborn as sn
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing,tree,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, mean_squared_error,explained_variance_score
from sklearn import datasets, linear_model
from sklearn.feature_selection import VarianceThreshold,RFE
import joblib 


# In[2]:


img=cv2.imread('test.tiff',cv2.IMREAD_UNCHANGED)


# In[3]:


file='INPUT/S2B_tile_20180312_18NXM_0_(B03-B08)_(B03+B08)_M_[-73.22078704833986,5.5807922489019255,-73.16559791564943,5.628200566956353].tiff'

def describe(file_name,show=False):

    # to open a tiff file for reading:
    tif = TIFF.open(file_name, mode='r')
    # to read all images in a TIFF file:
    for image in tif.iter_images(): # do stuff with image
    
        imggray=image[:,:,0]
        color = ('b','g','r')
        features=[]
        for i,col in enumerate(color):
            histr = cv2.calcHist([image],[i],None,[256],[0,256])
            features=features+(histr[:,0].tolist())
            if show:
                plt.plot(histr,color = col)
                plt.xlim([0,256])
        if show:
            plt.show()
        ret2,th2 = cv2.threshold(imggray,20,255,cv2.THRESH_BINARY_INV)
        #ret2,th2 = cv2.threshold(image[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        #cv2.imwrite('Input.jpg',image)
        #cv2.imwrite('Gray.jpg',imggray)
        #cv2.imwrite('BinaryImage.jpg',th2)
    
        contours,hierarchy = cv2.findContours(th2,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(len(contours))
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        areabig=cv2.contourArea(biggest_contour)
        M = cv2.moments(biggest_contour)
        #print (M)

        #print(areabig)
        result=areabig*25
        #print(result)
        resultkm2=result/1000000
        #print(resultkm2)
        features=features+[resultkm2]
        features=features+(list(M.values()))
        image_external = np.zeros(image.shape, image.dtype)
        cv2.drawContours(image_external,  biggest_contour, -1, (0,0,255), 10)
        
        if show:
            plt.imshow(image)
            plt.title('Test Image'), plt.xticks([]), plt.yticks([])
            plt.show()    
            plt.imshow(imggray, cmap='gray')
            plt.title('Test Image gray'), plt.xticks([]), plt.yticks([])
            plt.show()
            plt.imshow(th2, cmap='gray')
            plt.title('Threshold'), plt.xticks([]), plt.yticks([])
            plt.show()
            plt.imshow(cv2.cvtColor(image_external, cv2.COLOR_BGR2RGB))
            plt.title('contour'), plt.xticks([]), plt.yticks([])
            plt.show()
        cv2.imwrite('Gray.jpg',imggray)
        cv2.imwrite('Th2.jpg',th2)
        cv2.imwrite('Contours.jpg',image_external)
        return features
        
        

features=describe(file,False)
print(features,size(features))


# In[4]:


import os
import datetime
sourcedir = 'INPUT'

dl = os.listdir(sourcedir)

#print(dl)
data=[]
dates=[]


for file in dl:
    
    data.append(describe(sourcedir+'/'+file,False))
    result=file.split('_')
    #print(result)
    date=''
    if result[0]=='LC08':
        date1=datetime.datetime.strptime(result[3], '%Y%m%d').strftime('%Y-%m-%d')
        
    else:
        date1=datetime.datetime.strptime(result[2], '%Y%m%d').strftime('%Y-%m-%d')
    dates.append(date1)

print(dates)


# In[5]:


import datetime
import pandas as pd
#xlrd library requirement

def check_values(fecha):

    sourcedir = 'datos'
    fecha_split=fecha.split('-')
    year=fecha_split[0]
    month=fecha_split[1]
    day=fecha_split[2]
    
    fecha_dict={'01':'Enero','02':'Febrero','03':'Marzo','04':'Abril','05':'Mayo','06':'Junio','07':'Julio','08':'Agosto','09':'Septiembre','10':'Oct','11':'Nov','12':'Dic'}

    dl = os.listdir(sourcedir)

    for file in dl:
        if year in file and '.~' not in file:
            xls = pd.ExcelFile(sourcedir+'/'+file)
            sheet=xls.parse(fecha_dict[month],header=3)
            print(file,month)
            for i in range(len(sheet['FECHA'])):
                if fecha in str(sheet['FECHA'][i]):
                    #print('encontrado')
                    #print(sheet['NIVEL COTA m.s.n.m'][i]) 
                    return sheet['NIVEL COTA m.s.n.m'][i]
    
print("trabajando...")
labels =[]
for date in dates:
    labels.append(round(check_values(date)-2600,6))
print(labels)

#print(data)

    
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(data)
sel = VarianceThreshold(threshold=0.085)
data_scaled_in=list(data_scaled)
sel.fit(data_scaled_in)
data_sav=sel.transform(data_scaled_in)
print(data_sav.shape,data_scaled.shape)

tags=[]
for i in range (len(data_sav[0])):
    tags.append("features "+str(i))

#print(data_scaled[0])

featuresdf = pd.DataFrame(data_sav,columns=tags)
labelsdf=pd.DataFrame(labels,columns=["Label"])
                                    
df = pd.concat([featuresdf, labelsdf], axis=1)
df.to_csv("dataset.csv",index=False)
print("listo")


# In[6]:


rdata=pd.read_csv("dataset.csv")
rdata = rdata.sort_values('Label')

'''Your code here'''

print(rdata.describe())


# In[7]:



data=rdata.iloc[np.random.permutation(len(rdata))] # baraja los datos por filas



cut=int(len(data['Label'])*0.9)

train = data[:cut]
test_ = data[cut:]

trdata = train.copy()
del trdata['Label']


tstdata = test_.copy()
del tstdata['Label']

results={}


# In[8]:


target=train['Label']
expected=test_['Label']
regr = linear_model.LinearRegression()

regr.fit(trdata, target)

# Make predictions using the testing set
_pred = regr.predict(tstdata)
print(_pred)
print(list(expected))
print('abs error')
print(list(expected)-_pred)
print(mean_squared_error(expected, _pred))
print(explained_variance_score(expected, _pred))

result={}
result['mean_squared']=mean_squared_error(expected, _pred)
result['evs']=explained_variance_score(expected, _pred)

results['linear_regressor']=result


# In[9]:


target=train['Label']
expected=test_['Label']
regr =linear_model.Ridge(alpha=.5)

regr.fit(trdata, target)

# Make predictions using the testing set
_pred = regr.predict(tstdata)
print(_pred)
print(list(expected))
print('abs error')
print(list(expected)-_pred)
print(mean_squared_error(expected, _pred))
print(explained_variance_score(expected, _pred))

result={}
result['mean_squared']=mean_squared_error(expected, _pred)
result['evs']=explained_variance_score(expected, _pred)

results['ridge_linear_regressor']=result


# In[10]:


target=train['Label']
expected=test_['Label']
regr = svm.SVR()

regr.fit(trdata, target)

# Make predictions using the testing set
_pred = regr.predict(tstdata)
print(_pred)
print(list(expected))
print('abs error')
print(list(expected)-_pred)
print(mean_squared_error(expected, _pred))
print(explained_variance_score(expected, _pred))

result={}
result['mean_squared']=mean_squared_error(expected, _pred)
result['evs']=explained_variance_score(expected, _pred)

results['SVR']=result


# In[11]:


target=train['Label']
expected=test_['Label']
regr = MLPRegressor(solver='adam',activation='identity',hidden_layer_sizes=(120,60),max_iter=3000)

regr.fit(trdata, target)

# Make predictions using the testing set
_pred = regr.predict(tstdata)
print(_pred)
print(list(expected))
print('abs error')
print(list(expected)-_pred)
print(mean_squared_error(expected, _pred))
print(explained_variance_score(expected, _pred))

result={}
result['mean_squared']=mean_squared_error(expected, _pred)
result['evs']=explained_variance_score(expected, _pred)

results['mlpr']=result


# In[12]:


# try with another value of RFE
target=train['Label']
expected=test_['Label']

lm = linear_model.LinearRegression()
lm.fit(trdata, target)

rfe = RFE(lm, n_features_to_select=6)             
rfe = rfe.fit(trdata, target)

# predict prices of X_test
_pred = rfe.predict(tstdata)
print(_pred)
print(list(expected))
print('abs error')
print(list(expected)-_pred)
print(mean_squared_error(expected, _pred))
print(explained_variance_score(expected, _pred))

result={}
result['mean_squared']=mean_squared_error(expected, _pred)
result['evs']=explained_variance_score(expected, _pred)

results['lmrfe']=result


# In[13]:


target=train['Label']
expected=test_['Label']
regr =linear_model.Ridge(alpha=.5)

regr.fit(trdata, target)

rfe = RFE(regr, n_features_to_select=6)             
rfe = rfe.fit(trdata, target)

# predict prices of X_test
_pred = rfe.predict(tstdata)
print(_pred)
print(list(expected))
print('abs error')
print(list(expected)-_pred)
print(mean_squared_error(expected, _pred))
print(explained_variance_score(expected, _pred))

result={}
result['mean_squared']=mean_squared_error(expected, _pred)
result['evs']=explained_variance_score(expected, _pred)

results['lmridgerfe']=result



# In[14]:

file_object = open('sample.txt', 'a')
# Append 'hello' at the end of file
file_object.write(str(results))
# Close the file
file_object.close()






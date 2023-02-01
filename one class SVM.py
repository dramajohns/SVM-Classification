# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 11:07:33 2022

@author: fedib
"""

""" classification """
import pandas as pd
from sklearn import svm
import numpy as np

###### Q1,2,3,4
Dataset = pd.read_csv('C:/STUDY/SEMESTRE 2/machine learning/tp/tp1/diabetes.csv')

dataset_selected1= Dataset.loc[Dataset ['Outcome'].isin([1])] ## extraire les donner de label 1
dataset_selected0= Dataset.loc[Dataset ['Outcome'].isin([0])] ## extraire les donner de label 0

label_1=dataset_selected1['Outcome']
data1=dataset_selected1.drop(['Outcome'],axis=1)

label_0=dataset_selected0['Outcome']-1
data0=dataset_selected0.drop(['Outcome'],axis=1)
##########
#### Q5 

##### Q6

from sklearn.model_selection import train_test_split 
x_train1,x_test1,train_label,test_label=train_test_split(data1,label_1,test_size = 0.33,random_state = 0)


model=svm.OneClassSVM(kernel='linear', nu=0.1)
model.fit(x_train1)#train/apprentissage
test_data=np.concatenate((x_test1,data0),axis=0)
test_label_total=np.concatenate((test_label,label_0),axis=0)
pred=model.predict(test_data)

#######
"""calcule accuracy"""
########

from sklearn.metrics import accuracy_score

ACC=accuracy_score(test_label_total, pred)*100
print(ACC)
"""auc"""
from sklearn.metrics import roc_curve, auc
fp, tp, threshold = roc_curve(test_label_total,pred,pos_label=1)
print(fp,tp)
AUC=auc(fp,tp)*100
print(AUC)

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(test_label_total, pred)
print(CM)

#######
from sklearn.metrics import classification_report
print(classification_report(test_label_total, pred))

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(test_label_total, pred)
print(CM)



import matplotlib.pyplot as plt
plt.plot(fp, tp, color='blue',label = 'AUC = %0.2f' % AUC)
plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % AUC)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


########
""" visualizing confusion matrix using heatmap"""
###############

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
# CM1=pd.DataFrame(CM)
# print(CM1)
sns.heatmap(pd.DataFrame(CM), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
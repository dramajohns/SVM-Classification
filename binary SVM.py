# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:46:43 2022

@author: fedib
"""

label.value_counts()


""" classification """
import pandas as pd
from sklearn import svm
import numpy as np

dataset = pd.read_csv('C:/STUDY/SEMESTRE 2/machine learning/tp/tp1/diabetes.csv')

print(dataset)
dataset.info()
dataset.describe()

label=dataset['Outcome']
data=dataset.drop(['Outcome'],axis=1)

"""devision de la base de donnees en 2/3 et 1/3 """

from sklearn.model_selection import train_test_split 
x_train1,x_test1,train_label,test_label=train_test_split(data,label,test_size = 0.33,random_state = 0)

########
"""2/3_data_train,1/3_data_test, 2/3_label_train, 1/3_label_test=train_test_split()"""
########

from sklearn import svm
#SUPPORT  VECTOR MACHINE

model=svm.SVC(kernel='linear',C=1)
model.fit(x_train1,train_label)#train/apprentissage
pred=model.predict(x_test1)

#######
"""calcule accuracy"""
########

from sklearn.metrics import accuracy_score

ACC=accuracy_score(test_label, pred)*100
print(ACC)


from sklearn.metrics import confusion_matrix
CM = confusion_matrix(test_label, pred)
print(CM)

#######
from sklearn.metrics import classification_report
print(classification_report(test_label, pred))

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(test_label, pred)
print(CM)



import time
debut=time.time()
model.fit(x_train1,train_label)# train/apprentissage
temp=time.time()-debut
print(temp)


""" auc"""
from sklearn.metrics import roc_curve, auc
fp, tp, threshold = roc_curve(test_label,pred,pos_label=1)
print(fp,tp)
AUC=auc(fp,tp)*100
print(AUC)


#########
"""roc curve"""
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
#########


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







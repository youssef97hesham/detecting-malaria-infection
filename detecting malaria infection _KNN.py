# -*- coding: utf-8 -*-
"""
Created on Sat May 11 17:07:01 2019

@author: Youssef
"""

#import graphviz 
#from sklearn.tree import export_graphviz
import os 
import numpy as np
from PIL import Image 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
 
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    print("nnnnnnnnnnnnnnnnn")
    labels = []
    images = []
    c= 0
    s=16
    for d in directories:
        
        label_directory = os.path.join(data_directory, d)
        
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".png")]
        
        for f in file_names:
            c= c+1
            print(c)
            i = Image.open(f).convert('L')
            print(i)
            x = i.resize((s,s))
            print (x)
            i = np.array(x).reshape(s*s)
            print(i.shape)
            images.append(i)
            labels.append(d)
    return images, labels


images, labels = load_data(os.path.join("cell_images"))
X = np.array(images)
y = np.array(labels)

print("start split")
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.3, random_state=21 , stratify=y)
ParamGrid = {'n_neighbors':np.arange(1,10)}
knn = KNeighborsClassifier()
knnCV = GridSearchCV(knn , ParamGrid ,cv=5)
print("start fit")
knnCV.fit(X_train, y_train)
newparm = knnCV.best_params_
newparmvalue = newparm.get('n_neighbors')
print(newparm)
print(newparmvalue)
knnnew = KNeighborsClassifier(n_neighbors=newparmvalue)
print("start new fit")
knnnew.fit(X_train, y_train)
print("start new predict")
y_pred = knnnew.predict(X_test)
knnnew.score(X_test, y_test)
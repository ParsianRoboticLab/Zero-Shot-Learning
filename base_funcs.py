from sklearn import tree
import numpy as np
from params import *


def load_label(file_name):
    with open(file_name) as labels:
        content = labels.read().split('\n')
    label_list = [x.split('\t')[0] for x in content]
    return label_list

def load_label_attr(file_name):
    with open(file_name) as labels:
        content = labels.read().split('\n')
    label_attr = [list(map(float,x.split('\t')[1:])) for x in content]
    atrs = np.array(label_attr[0])
    label_attr = label_attr[1:]
    for i in label_attr:
        ii = np.array(i)
        atrs = np.vstack((atrs,ii))
    return atrs

def dsc_tree_train(X,y):
    attr_clf = tree.DecisionTreeClassifier()
    attr_clf.fit(X,y)

    return attr_clf

def dsc_tree_prediction(clf,X):
    return clf.predict(X)

def create_label_to_atr_dict():
    global labels_to_atrs_dict
    labels_list = load_label('attributes_per_class.txt')
    atrs_list = load_label_attr('attributes_per_class.txt')
    for label in labels_list:
        labels_to_atrs_dict[label] = atrs_list[labels_list.index(label)]

def label_to_atr(label):
    return labels_to_atrs_dict[label]

def atr_to_label(atr):
    for key,value in labels_to_atrs_dict.items():
        if set(value) == set(atr):
            return key
    tree = dsc_tree_train(load_label_attr('attributes_per_class.txt'),load_label('attributes_per_class.txt'))
    return dsc_tree_prediction(tree,atr)

# def write_submition(img_name,label,write_in_new_file_flag):



    



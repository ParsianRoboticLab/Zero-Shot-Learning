from sklearn import tree
import numpy as np
from PIL import Image
import pickle
import torch.nn.init as init

labels_to_atrs_dict = {}
train_pics_dict={}
train_pics_to_attr_dict = {}
test_pics_list = []
readable_to_label_dict = {}


def load_label(file_name):
    with open(file_name) as labels:
        content = labels.read().split('\n')
    label_list = [x.split('\t')[0] for x in content][:-1]
    return label_list

def load_label_attr(file_name):
    with open(file_name) as labels:
        content = labels.read().split('\n')
    label_attr = [list(map(float,x.split('\t')[1:])) for x in content][:-1]
    atrs = np.array(label_attr[0])
    label_attr = label_attr[1:]
    for i in label_attr:
        ii = np.array(i)
        atrs = np.vstack((atrs,ii))
    return atrs

def create_label_to_atr_dict():
    labels_list = load_label('attributes_per_class.txt')
    atrs_list = load_label_attr('attributes_per_class.txt')
    for label in labels_list:
        labels_to_atrs_dict[label] = atrs_list[labels_list.index(label)]

def label_to_atr(label):
    return labels_to_atrs_dict[label]
import math
def newAttr_to_label(atr):
    maxProduct = 0;
    bestKey = 'ZJL1'
    for key,value in labels_to_atrs_dict.items():
        p = 1
        for i in range(len(value)):
            p = p * (math.fabs(atr[i] - value [i]))
        if p > maxProduct :
            maxProduct = p
            bestKey = key
    return bestKey

def atr_to_label(atr):
    for key,value in labels_to_atrs_dict.items():
        try:
            if set(value) == set(atr):
                return key
        except:
            print('key is:', key, 'value shape is:', value.shape, 'atr shape is:', atr.shape)
    tree = dsc_tree_train(load_label_attr('attributes_per_class.txt'),load_label('attributes_per_class.txt'))
    tst_atr = np.array([atr])
    return dsc_tree_prediction(tree,tst_atr)


uniques = ['ZJL296', 'ZJL298', 'ZJL301', 'ZJL302', 'ZJL303', 'ZJL312', 'ZJL320', 'ZJL322', 'ZJL326', 'ZJL330', 'ZJL349', 'ZJL351', 'ZJL352', 'ZJL354', 'ZJL361', 'ZJL364', 'ZJL377', 'ZJL393', 'ZJL395', 'ZJL404', 'ZJL406', 'ZJL408', 'ZJL410', 'ZJL411', 'ZJL416', 'ZJL420', 'ZJL422', 'ZJL431', 'ZJL434', 'ZJL441', 'ZJL447', 'ZJL451', 'ZJL462', 'ZJL469', 'ZJL474', 'ZJL481', 'ZJL485', 'ZJL486', 'ZJL489', 'ZJL491', 'ZJL496', 'ZJL497', 'ZJL498', 'ZJL499', 'ZJL500']
def simpleDap(atr):
    maxProduct = 0;
    bestKey = 'ZJL1'
    for key,value in labels_to_atrs_dict.items():
        for un in uniques:
            if key == un:
                continue
        p = 1
        for i in range(len(value)):
            p = p * (1 - ((math.fabs(atr[i] - value [i])))+0.00000001)
        if p > maxProduct :
            maxProduct = p
            bestKey = key
    return bestKey

def create_train_pics_dict():
    file_name = "train.txt"
    with open(file_name) as pics_labels:
        content = pics_labels.read().split('\n')
    pics_list = [x.split('\t')[0] for x in content][:-1]
    lables_list = [x.split('\t')[1:] for x in content][:-1]
    for pic in pics_list:
        train_pics_dict[pic] = lables_list[pics_list.index(pic)][0]
def create_test_pics_list():
    file_name = "image.txt"
    with open(file_name) as pics_labels:
        a = pics_labels.read().split('\n')[:-1]
        test_pics_list.extend(a)
def load_pic(pic_name,train_flag):
    if train_flag:
        pic = "train/"+pic_name
    else:
        pic = "test/"+pic_name
    return Image.open(pic)

def create_train_pic_atr_dict():
    for pic,label in train_pics_dict.items():
        train_pics_to_attr_dict[pic] = labels_to_atrs_dict[label]


def dsc_tree_train(X,y):
    attr_clf = tree.DecisionTreeClassifier()
    attr_clf.fit(X,y)
    return attr_clf

def dsc_tree_prediction(clf,X):
    return clf.predict(X)

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

seen_readable_to_label_dict = {}
unseen_readable_to_label_dict = {}
def create_readable_to_label():
    global readable_to_label_dict
    file_name = "label_list.txt"
    with open(file_name) as pics_labels:
        content = pics_labels.read().split('\n')[:-1]
    labels_list = [x.split('\t')[0] for x in content]
    readables_list = [x.split('\t')[1] for x in content]
    for readable in readables_list:
        readable_to_label_dict[readable] = labels_list[readables_list.index(readable)]

def create_seen_readable_to_label():
    global train_pics_dict, seen_readable_to_label_dict
    try:
        train_pics_dict = np.load('train_pics_dict.npy')[()]
    except:
        create_train_pics_dict()
        np.save('train_pics_dict.npy',train_pics_dict)
    file_name = "label_list.txt"
    with open(file_name) as pics_labels:
        content = pics_labels.read().split('\n')[:-1]
    seen_labels_list = train_pics_dict.values()
    labels_list = [x.split('\t')[0] for x in content]
    readables_list = [x.split('\t')[1] for x in content]
    for label in seen_labels_list:
        seen_readable_to_label_dict[readables_list[labels_list.index(label)]] = label

def create_unseen_readable_to_label():
    global train_pics_dict, unseen_readable_to_label_dict
    try:
        train_pics_dict = np.load('train_pics_dict.npy')[()]
    except:
        create_train_pics_dict()
        np.save('train_pics_dict.npy',train_pics_dict)
    file_name = "label_list.txt"
    with open(file_name) as pics_labels:
        content = pics_labels.read().split('\n')[:-1]
    seen_labels_list = train_pics_dict.values()
    labels_list = [x.split('\t')[0] for x in content]
    readables_list = [x.split('\t')[1] for x in content]
    for label in labels_list:
        if label not in seen_labels_list:
            unseen_readable_to_label_dict[readables_list[labels_list.index(label)]] = label

def initialize():
    global labels_to_atrs_dict, train_pics_to_attr_dict, train_pics_dict, dcs_tree,readable_to_label_dict,test_pics_list, seen_readable_to_label_dict, unseen_readable_to_label_dict
    try:
        labels_to_atrs_dict = np.load('labels_to_atrs_dict.npy').items()
    except:
        create_label_to_atr_dict()
        np.save('labels_to_atrs_dict.npy',labels_to_atrs_dict)
    try:
        tr = np.load('train_pics_dict.npy').items()
        train_pics_dict = tr
    except:
        create_train_pics_dict()
        np.save('train_pics_dict.npy',train_pics_dict)
    try:
        train_pics_to_attr_dict = np.load('train_pics_to_attr_dict.npy').items()
    except:
        create_train_pic_atr_dict()
        np.save('train_pics_to_attr_dict.npy',train_pics_to_attr_dict)
    try:
        readable_to_label_dict = np.load('readable_to_label_dict.npy').items()
    except:
        create_readable_to_label()
        np.save('readable_to_label_dict.npy',readable_to_label_dict)
    try:
        seen_readable_to_label_dict = np.load('seen_readable_to_label_dict.npy').items()
    except:
        create_seen_readable_to_label()
        np.save('seen_readable_to_label_dict.npy',seen_readable_to_label_dict)
    try:
        unseen_readable_to_label_dict = np.load('unseen_readable_to_label_dict.npy').items()
    except:
        create_unseen_readable_to_label()
        np.save('unseen_readable_to_label_dict.npy',unseen_readable_to_label_dict)
    create_test_pics_list()
# def write_submition(img_name,label,write_in_new_file_flag):

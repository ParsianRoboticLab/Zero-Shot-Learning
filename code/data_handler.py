import os
import numpy as np
from base_funcs import *
from PIL import Image
initialize()
train_pics_dict = np.load('train_pics_dict.npy')[()]
labels_to_atrs_dict = np.load('labels_to_atrs_dict.npy')[()]
train_pics_to_attr_dict = np.load('train_pics_to_attr_dict.npy')[()]
readable_to_label_dict = np.load('readable_to_label_dict.npy')[()]
seen_readable_to_label_dict = np.load('seen_readable_to_label_dict.npy')[()]
unseen_readable_to_label_dict = np.load('unseen_readable_to_label_dict.npy')[()]


class DataHandler:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.test_data_loaded = False

    def load_data(self):

        self.all_classes = {}
        self.all_classes_r = {}
        count = 0
        for k, v in readable_to_label_dict.items():
            self.all_classes[v] = count
            self.all_classes_r[count] = v
            count += 1
        #
        self.test_classes = []
        self.train_classes = []
        for k,v in self.all_classes.items():
            self.test_classes.append(v)
            self.train_classes.append(v)

        self.all_attr = np.zeros([0, 50])
        for k, v in labels_to_atrs_dict.items():
            self.all_attr = np.vstack((self.all_attr,v))

        # Train files load
        self.load_train_data()
        self.load_test_data()
        print('done')

    def preprocess_data(self):

        print('Preprocess')
        
        # Do everything here so i can remove this function if i want to
        # Preprocess c and x
        self.epsilon = 1e-6
        self.attr_mean = np.mean(self.all_attr, axis=0, keepdims=True) # Note all_attr (Seen and Unseen classes) available at training time
        self.attr_std = np.std(self.all_attr, axis=0, keepdims=True)

        print(self.all_attr.shape)
        print(self.attr_mean.shape)
        print(self.attr_std.shape)

        self.train_attr = np.divide(self.train_attr - self.attr_mean,  (self.attr_std + self.epsilon))
        self.all_attr = np.divide(self.all_attr - self.attr_mean,  (self.attr_std + self.epsilon))
        self.test_attr = np.divide(self.test_attr - self.attr_mean,  (self.attr_std + self.epsilon))
        
        self.train_data_mean = np.mean(self.train_data, axis=0, keepdims=True)
        self.train_data_std = np.std(self.train_data, axis=0, keepdims=True)

        self.train_data = np.divide(self.train_data - self.train_data_mean,  (self.train_data_std + self.epsilon))
        # Here only preprocessing the test data
        # Note: Test data has not been used for preprocessing (calculation of mean or std)
        self.test_data = np.divide(self.test_data - self.train_data_mean,  (self.train_data_std + self.epsilon))
        
    def load_train_data(self):

        self.train_label = np.zeros(0)
        for k, v in train_pics_dict.items():
            self.train_label = np.hstack((self.train_label, self.all_classes[v]))

        train_array = None
        attr_array = None
        label_array = None
        file_counter = 1
        train_dir = '../data/DatasetB/train'
        for (dirpath, dirname, filenames) in os.walk(train_dir):
            total_file = len(filenames)
            for filename in filenames:
                if file_counter % 100 == 0:
                    print('{} of {} Loaded'.format(file_counter, total_file))
                file_counter += 1
                image_data = Image.open(os.path.join(dirpath, filename)).convert('RGB')
                image_data = image_data.resize((64, 64))
                image_data = np.array(image_data).ravel().reshape(1, -1)
                image_attrs = np.reshape(labels_to_atrs_dict[train_pics_dict[filename]], (1, -1))
                image_label = self.all_classes[train_pics_dict[filename]]
                if train_array is None:
                    train_array = image_data
                    attr_array = image_attrs
                    label_array = image_label
                else:
                    train_array = np.vstack((train_array, image_data))
                    attr_array = np.vstack((attr_array, image_attrs))
                    label_array = np.vstack((label_array, image_label))
            self.train_data = train_array
            self.train_attr = attr_array
            self.train_label = label_array

            self.test_data = train_array
            self.test_attr = attr_array
            self.test_label = label_array


        self.train_size = self.train_data.shape[0]
        self.x_dim = self.train_data.shape[1]
        self.attr_dim = self.train_attr.shape[1]
        self.test_size = self.test_data.shape[0]

        print('Training Data: ' + str(self.train_data.shape))
        print('Training Attr: ' + str(self.train_attr.shape))

        print('Testing Data: ' + str(self.test_data.shape))
        print('Testing Attr: ' + str(self.test_attr.shape))
        print('Testing Classes' + str(len(self.test_classes)))
        print('Testing Labels' + str(self.test_label.shape))
        
    def load_test_data(self):
        pass

    def next_train_batch(self, index, batch_size):
        start_index = index
        end_index = index+batch_size
        return self.train_data[start_index:end_index], self.train_attr[start_index:end_index]

    def next_test_batch(self, index, batch_size):
        start_index = index
        end_index = index+batch_size
        return self.test_data[start_index:end_index], self.test_attr[start_index:end_index]

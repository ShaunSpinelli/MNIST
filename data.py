import pandas as pd
import numpy as np

class Data:

    def __init__(self, data_path):
        self.data_path = data_path
        self.arr = None
        self.train_set = None
        self.test_set = None

        self.img_shape = None
        self.img_size = None
        self.num_classes = None
        self.num_channels = None


    @staticmethod
    def load_csv(data_path):
        return pd.read_csv(data_path)

    @staticmethod
    def get_data_arr(df):
        """returns row of each number as list item"""
        arr = [df.iloc[i].values for i in range(len(df))]
        return  arr

    @staticmethod
    def reshape_images(arr):
        processed = []
        for image in arr:
            processed.append(image[1:])
            # processed.append(image[1:].reshape(28,28))
        return processed

    @staticmethod
    def create_labels_arr(arr):
        labels =  [i[0] for i in arr]
        zeros = np.zeros((len(arr), 10))
        zeros[np.arange(len(arr)), labels] = 1
        return zeros

    def process_data(self,):
        df = self.load_csv(self.data_path)
        self.arr = self.get_data_arr(df)

    def get_img_labls(self, arr):
        images = np.asarray(self.reshape_images(arr))
        labels = self.create_labels_arr(arr)
        return images, labels

    def split_data(self, test_size=0.2):
        '''splits data into test and train'''
        if self.arr == None:
            self.process_data()
        rand_idxs = np.random.randint(0, len(self.arr),len(self.arr))
        split = int(len(self.arr)*test_size)
        train_idxs = rand_idxs[split:]
        test_idxs = rand_idxs[:split]
        self.train_set = [self.arr[i] for i in train_idxs]
        self.test_set = [self.arr[i] for i in test_idxs]

    def shuffle(self):
        np.random.shuffle(self.train_set)



    def random_batch(self, bs, epoch, is_test=False ):

        if self.train_set is None:
            self.split_data()

        if is_test:
            data = self.get_img_labls(arr = self.test_set) #NOTE: these needs to be tuple of pic and labels
        else:
            data = self.get_img_labls(arr=self.train_set)

        if bs*epoch < len(data[0]):
            batch = data[0][bs*epoch:bs*epoch+bs], data[1][bs*epoch:bs*epoch+bs]
        else:
            #need to reshuffle self.data for train
            batch = data[0][:bs] , data[1][:bs]

        return batch



    # def get_images(self):
    #     if self.images == None:
    #         self.process_data(self.data_path)
    #     return self.images

    # def get_labels(self):
    #     if self.labels == None:
    #         self.process_data(self.data_path)
    #     return self.labels




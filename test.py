import pandas as pd
import numpy as np

class Data:

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None



    @staticmethod
    def load_csv(data_path):
        return pd.read_csv(data_path)

    @staticmethod
    def get_data_arr(df):
        """returns row of each number as list item"""
        arr = [df.iloc[i].values for i in range(len(df))]
        return  arr

    def process_data(self,):
        df = self.load_csv(self.data_path)
        self.data = self.get_data_arr(df)

    def batch(self, bs,)

        if self.data is None:
            self.process_data()

        data = self.get_img_labls(arr = self.test_set) #NOTE: these needs to be tuple of pic and labels


        return batch
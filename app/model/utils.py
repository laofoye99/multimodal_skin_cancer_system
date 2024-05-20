import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class Utils:
    def __init__(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'archive')
        self.meta_data = self.load_data()
        self.image_path = self.get_image_path()
    
    def get_image_path(self):
        image_path = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    image_path.append(os.path.abspath(file_path))
        return image_path
    
    def image_id_to_path(self):
        pattern = {}
        for path in self.image_path:
            if os.path.basename(path) in self.meta_data['img_id'].values:
                pattern[os.path.basename(path)] = path
        return pattern

    def load_data(self):
        metadata = pd.read_csv(os.path.join(self.data_path, 'metadata.csv'))
        metadata.drop(columns=['lesion_id', 'biopsed', 'patient_id'], inplace=True)
        metadata['label'] = metadata['diagnostic']
        metadata.drop(columns=['diagnostic'],inplace=True)
        return metadata
    
    def data_process(self):
        self.meta_data['img_id'] = self.meta_data['img_id'].replace(self.image_id_to_path())
        self.metadata = self.encode()
        return self.meta_data

    def encode(self):
        self.meta_data.replace({'UNK':-1, 'NaN':-1, np.nan:-1}, inplace=True)
        self.meta_data['smoke'] = self.meta_data['smoke'].replace({'False':0, 'True':1}).astype(int)
        self.meta_data['drink'] = self.meta_data['drink'].replace({'False':0, 'True':1}).astype(int)
        self.meta_data['background_father'] = self.meta_data['background_father'].replace({'AUSTRIA':0, 'BRASIL':1, 'BRAZIL':2, 'CZECH':3, 
                                                                                           'GERMANY':4, 'ISRAEL':5, 'ITALY':6, 'NETHERLANDS':7, 
                                                                                           'POLAND':8, 'POMERANIA':9, 'PORTUGAL':10, 'SPAIN':11})
        self.meta_data['background_mother'] = self.meta_data['background_mother'].replace({'BRAZIL':0, 'FRANCE':1, 'GERMANY':2, 'ITALY':3, 
                                                                                           'NETHERLANDS':4, 'NORWAY':5, 'POLAND':6, 'POMERANIA':7, 
                                                                                           'PORTUGAL':8, 'SPAIN':9})
        self.meta_data['pesticide'] = self.meta_data['pesticide'].replace({'False':0, 'True':1}).astype(int)
        self.meta_data['gender'] = self.meta_data['gender'].replace({'FEMALE':0, 'MALE':1})
        self.meta_data['skin_cancer_history'] = self.meta_data['skin_cancer_history'].replace({'False':0, 'True':1}).astype(int)
        self.meta_data['cancer_history'] = self.meta_data['cancer_history'].replace({'False':0, 'True':1}).astype(int)
        self.meta_data['has_piped_water'] = self.meta_data['has_piped_water'].replace({'False':0, 'True':1}).astype(int)
        self.meta_data['has_sewage_system'] = self.meta_data['has_sewage_system'].replace({'False':0, 'True':1}).astype(int)
        self.meta_data['region'] = self.meta_data['region'].replace({'ABDOMEN':0, 'ARM':1, 'BACK':2, 'CHEST':3, 
                                                                     'EAR':4, 'FACE':5, 'FOOT':6, 'FOREARM':7, 
                                                                     'HAND':8, 'LIP':9, 'NECK':10, 'NOSE':11, 
                                                                     'SCALP':12, 'THIGH':13})
        self.meta_data['itch'] = self.meta_data['itch'].replace({'FALSE':0, 'TRUE':1}).astype(int)
        self.meta_data['grew'] = self.meta_data['grew'].replace({'FALSE':0, 'TRUE':1}).astype(int)
        self.meta_data['hurt'] = self.meta_data['hurt'].replace({'FALSE':0, 'TRUE':1}).astype(int)
        self.meta_data['changed'] = self.meta_data['changed'].replace({'FALSE':0, 'TRUE':1}).astype(int)
        self.meta_data['bleed'] = self.meta_data['bleed'].replace({'FALSE':0, 'TRUE':1}).astype(int)
        self.meta_data['elevation'] = self.meta_data['elevation'].replace({'FALSE':0, 'TRUE':1}).astype(int)
        self.meta_data['label'] = self.meta_data['label'].replace({'ACK':0, 'BCC':1, 'MEL':2, 'NEV':3, 'SCC':4, 'SEK':5})

    def train_test_split(self):
        X = self.meta_data.drop(columns=['label'])
        y = self.meta_data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    utils = Utils()
    df = utils.data_process()
    print(df.head())
    
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import NormalizeFeatures

import os

def load_data(dataset_name):
    if dataset_name == 'AIDS':
        path = os.path.join(os.getcwd(), 'data','AIDS')
        dataset = Planetoid(path,name='AIDS700nef')
        
        return dataset
    
    if dataset_name == 'LINUX':
        path = os.path.join(os.getcwd(), 'data','LINUX')
        dataset = Planetoid(path,name='LINUX')
        return dataset
    
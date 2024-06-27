import torch.utils.data
import numpy as np
import h5py
import random
from ruamel.yaml import YAML
from torch.utils.data import Dataset, DataLoader
import torch

class rcc_3d_dataset(Dataset):
    def __init__(self, list_id, config):
        self.config = config
        with h5py.File(config['path_data'], 'r', swmr=True) as f:
            start_indexes = []
            accumulated_count = 0
            new_list_id = []
            for id in list_id:
                try: 
                    shape_pre = f[id]['image']['precontrast_r']['Axial'].shape
                    shape_post = f[id]['image']['post_50sec']['Axial'].shape
                    shape_late = f[id]['image']['post_5min_r']['Axial'].shape
                    shape_mask = f[id]['segmentation']['post_50sec']['Axial'].shape
                except KeyError:
                    continue

                assert(shape_pre == shape_post == shape_late == shape_mask)
                start_indexes.append(accumulated_count)
                accumulated_count += (shape_pre[0]-1)//5
                new_list_id.append(id)

        self.list_id = new_list_id
        self.start_indexes = start_indexes ##
        self.tot_count = accumulated_count ##
        self.path_data = config['path_data']

    def __len__(self):
        return self.tot_count
    
    def __getitem__(self, idx):
        id_index = np.digitize(idx, self.start_indexes)-1
        slice_index = (idx - self.start_indexes[id_index])*5
        image_x = h5py.File(self.path_data, 'r', swmr=True)[self.list_id[id_index]]['image']['precontrast_r']['Axial'][slice_index:slice_index+1, :256, :256]
        image_y = h5py.File(self.path_data, 'r', swmr=True)[self.list_id[id_index]]['image']['post_50sec']['Axial'][slice_index:slice_index+1, :256, :256]

        image_x = (image_x/2000).astype(np.float32)
        image_y = (image_y/2000).astype(np.float32)
    
        #print(image_x.shape, image_y.shape)
        #tt = torch.from_numpy(np.stack([image_x, image_y], axis=0))
        #print(tt.shape)

        #return tt
        #return torch.from_numpy(image_x).squeeze(dim=0), torch.from_numpy(image_y).squeeze(dim=0)
        #return torch.from_numpy(image_x).unsqueeze(dim=1), torch.from_numpy(image_y).unsqueeze(dim=1)
        return torch.from_numpy(image_x), torch.from_numpy(image_y)

def CreateDatasetSynthesis(phase, input_path = None, contrast1 = 'T1', contrast2 = 'T2'):
    yaml = YAML()
    with open('/home/synergyai/jth/rcc-classification-research/experiments/config/session_240621.yaml') as f:
        config = yaml.load(f)

    split_dict = {}
    with open(config['path_split'],'r') as f_split:
        split = f_split.read().splitlines()
        for i, split_name in enumerate(['train', 'valid', 'test']):
            string = split[i]
            string = string.replace('[','').replace(']','').replace("'", "")
            id_list = string.split(', ')
            split_dict[split_name] = id_list
    
    if phase == 'train':
        list_id = split_dict['train']
    elif phase == 'val':
        list_id = split_dict['valid']

    return rcc_3d_dataset(list_id, config)
    

# def CreateDatasetSynthesis(phase, input_path, contrast1 = 'T1', contrast2 = 'T2'):

#     target_file = input_path + "/data_{}_{}.mat".format(phase, contrast1)
#     data_fs_s1=LoadDataSet(target_file)
    
#     target_file = input_path + "/data_{}_{}.mat".format(phase, contrast2)
#     data_fs_s2=LoadDataSet(target_file)

#     dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_s1),torch.from_numpy(data_fs_s2))  
#     return dataset 



#Dataset loading from load_dir and converintg to 256x256 
def LoadDataSet(load_dir, variable = 'data_fs', padding = True, Norm = True):
    f = h5py.File(load_dir,'r') 
    if np.array(f[variable]).ndim==3:
        data=np.expand_dims(np.transpose(np.array(f[variable]),(0,2,1)),axis=1)
    else:
        data=np.transpose(np.array(f[variable]),(1,0,3,2))
    data=data.astype(np.float32) 
    if padding:
        pad_x=int((256-data.shape[2])/2)
        pad_y=int((256-data.shape[3])/2)
        print('padding in x-y with:'+str(pad_x)+'-'+str(pad_y))
        data=np.pad(data,((0,0),(0,0),(pad_x,pad_x),(pad_y,pad_y)))   
    if Norm:    
        data=(data-0.5)/0.5      
    return data



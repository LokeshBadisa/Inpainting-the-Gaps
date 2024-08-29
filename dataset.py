from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pathlib import Path
from utils import multiload, getcombinations, synset_to_label, default_transform
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

class PartImageNetDataset2(Dataset):
    def __init__(self, annfile, type='fullout', device='cuda:2'):
        self.maskroot = Path(annfile).parent
        self.imageroot = self.maskroot.parent.parent/'images'/ Path('test')
        self.type = type
        if type == 'mask':
            self.modifiedroot = Path('../partset32/blackpatches')
        elif type == 'inpainted':
            self.modifiedroot = Path('../partset32/inpainted')
        elif type == 'fullout':
            self.modifiedroot = Path('../partset32/fullout')
        self.data = COCO(annfile)
        
        self.transform = default_transform          
        self.device = torch.device(device)


    def __len__(self):
        return len(self.data.dataset['images'])
    
    def __getitem__(self, idx):
        file_name = Path(self.data.loadImgs(idx)[0]['file_name'])
        image = Image.open(str(self.imageroot/file_name)).convert('RGB')
        image = self.transform(image)
        c = synset_to_label[file_name.stem.split('_')[0]]
        sample = {'image': image, 'class': int(c), 'index': idx}
        return sample

    #Only for Single Deletion  
    def get_interventions(self, idx):
        file_name = self.data.loadImgs(idx)[0]['file_name']
        parts = self.data.getAnnIds(imgIds=idx)
        L = {}

        if self.type != 'fullout':           
            for i in range(len(parts)):
                new_f = self.modifiedroot / Path(file_name.split('.')[0] + '_' + str(i).zfill(2) + '.png')
                if new_f.exists():
                    image = Image.open(new_f).convert('RGB')
                    image = self.transform(image).unsqueeze(0)
                    L[parts[i]] = image
            
        else:
            try:
                total_mask = multiload(self.data, parts)
            except:
                return L
            base_image = self.modifiedroot / Path(file_name.split('.')[0] + '.png')
            base_image = Image.open(base_image).convert('RGB')
            img = Image.open(self.imageroot / Path(file_name)).convert('RGB')
            base_image = np.array(base_image.resize(img.size))
            img = np.array(img)

            
            for i in range(len(parts)):       
                mask = total_mask - self.data.annToMask(self.data.loadAnns(parts[i])[0])
                mask[mask>=1] = 1
                mask = np.stack([mask]*3, axis=-1)                
                L[parts[i]] = self.transform(Image.fromarray(base_image * (1-mask) + img * mask)).unsqueeze(0)
                
        return L
                
                

    #For Deletion Check & Preservation Check
    def get_interventions2(self, idx, parts_to_remove, keys = None):
        file_name = self.data.loadImgs(idx)[0]['file_name']
        if keys is None:
            keys = self.get_p_thresholds()
        
        parts = self.data.getAnnIds(imgIds=idx)
        L = {}
        
        if self.type != 'fullout':
            
            combinations = getcombinations(parts)
            combinations = {j:i for i,j in enumerate(combinations)}           


            for threshold, each_partset in zip(keys, parts_to_remove):
                if each_partset != []:

                    if tuple(each_partset) not in combinations:                
                        continue
                    
                    key = combinations[tuple(each_partset)]
                    new_f = self.modifiedroot / Path(file_name.split('.')[0] + '_' + str(key).zfill(2) + '.png')
                    
                    # This if block is written because the partset requested
                    # might not be there in the images we have as we constrained
                    # ourselves to 32 images. 
                    if not new_f.exists():
                        continue
                    image = Image.open(new_f).convert('RGB')
                    
                # This else block is due to thresholding at higher value gives no parts.
                # So, removing no parts is same as original image.
                else:
                    image = Image.open(self.imageroot / Path(file_name)).convert('RGB')
                    
                image = self.transform(image).unsqueeze(0)
                L[threshold] = image

            # Above code is such that if you want to remove some parts then 
            # it should be in the images we have. Since we constrained the limit
            # to 32, we are bound to miss some images. To counter that,
            # we are using sending an empty list 
            # if len(L) !=0:
            #     L = torch.stack(L).squeeze()

        else:
            #This code is written for fullout images. Code here will 
            # such that behaviour is similar to the above code.
            try:
                total_mask = multiload(self.data, parts)
            except:
                return L
            base_image = self.modifiedroot / Path(file_name.split('.')[0] + '.png')
            base_image = Image.open(base_image).convert('RGB')
            img = Image.open(self.imageroot / Path(file_name)).convert('RGB')
            base_image = np.array(base_image.resize(img.size))
            img = np.array(img)

            for threshold, each_partset in zip(keys, parts_to_remove):
                if each_partset != []:
                    mask = total_mask - multiload(self.data, each_partset)
                    mask[mask>=1] = 1
                    mask = np.stack([mask]*3, axis=-1)     
                    image = base_image * (1-mask) + img * mask
                    image = Image.fromarray(image).convert('RGB')
                else:
                    image = Image.open(self.imageroot / Path(file_name)).convert('RGB')

                image = self.transform(image).unsqueeze(0)
                L[threshold] = image



        return L
    
    def get_incremental_interventions(self, idx, importances_dict, mode= 'positive'):
        total_area = self.data.loadImgs(idx)[0]['height'] * self.data.loadImgs(idx)[0]['width']
        
        areas = {}
        eff_importance = {}
        for i in importances_dict.keys():
            areas[i] = self.data.annToMask(self.data.loadAnns(i)[0]).sum()
            eff_importance[i] = importances_dict[i].item()/areas[i]
        mode = True if mode == 'positive' else False
        sorted_parts = sorted(eff_importance, key=eff_importance.get, reverse=mode)
        
        set_parts = []
        for i in range(1, len(sorted_parts)+1):
            set_parts.append(tuple(sorted(sorted_parts[:i])))

        intervened_images = self.get_interventions2(idx, set_parts, set_parts)
        areas = [sum([areas[j] for j in i]) for i in intervened_images.keys()]
        

        return intervened_images, areas, float(total_area)


    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)
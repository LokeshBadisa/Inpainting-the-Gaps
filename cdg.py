from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
from itertools import combinations
import json
import numpy as np
from argparse import ArgumentParser
from torchvision import transforms
from einops import rearrange    

data = COCO('PartImageNet/annotations/test/test.json')

i_t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def getcombinations(L,limit):
    ans = []
    for i in range(1,len(L)+1):
        ans += list(combinations(L,i))
        if len(ans) > limit:
            break
    if limit < len(ans):
        ans = ans[:limit]
    return ans

def multiload(data, L):
    ans = data.annToMask(data.loadAnns(L[0])[0])
    for i in range(1,len(L)):
        ans += data.annToMask(data.loadAnns(L[i])[0])
    return ans


#Inpainting generation
def generate_dataset(args):
    limit = args.limit
    exceptions = []
    Path(f'partset{limit}/masks').mkdir(exist_ok=True,parents=True)    
    Path(f'partset{limit}/images').mkdir(exist_ok=True,parents=True)
    labels = dict()

    for i in tqdm(data.getImgIds()):
        
        curr_anns = data.getAnnIds(imgIds=i)
        if len(curr_anns) < 2:
            continue
        
        
        stem = data.loadImgs(i)[0]['file_name'].split('.')[0]
        classname = stem.split('_')[0]
        img = Image.open(f'PartImageNet/images/test/{data.loadImgs(i)[0]["file_name"]}').resize((256,256))
        for k,j in enumerate(getcombinations(curr_anns,limit)):
            try:
                mask = multiload(data,j)
            except:
                exceptions.append(j)
                continue
            mask = np.where(mask>0,255,0).astype(np.uint8)
            mask = Image.fromarray(mask)
            mask = mask.resize((256,256))
            mask.save(f'partset{limit}/masks/{stem}_{k}.png')
            img.save(f'partset{limit}/images/{stem}_{k}.png')
            labels[f'{stem}_{k}.png'] = classname
    
    with open(f'partset{limit}/exceptions.txt','w') as f:
        for i in exceptions:
            f.write(str(i)+'\n')    
    with open(f'partset{limit}/labels.json','w') as f:
        json.dump(labels,f)


#Random Infilling helper
def random_infilling_helper(limit,index):
             
    curr_anns = data.getAnnIds(imgIds=index)
    if len(curr_anns) < 2:
        return
    
    labels = dict()
    
    stem = data.loadImgs(index)[0]['file_name'].split('.')[0]
    classname = stem.split('_')[0]
    img = Image.open(f'PartImageNet/images/test/{data.loadImgs(index)[0]["file_name"]}')
    img = np.array(img)
    if img.ndim < 3:
        return
    for k,j in enumerate(getcombinations(curr_anns,limit)):
        try:
            mask = multiload(data,j)
        except:
            continue
        mask = np.where(mask>0,1,0)
        a,b = mask.shape
        mask = np.stack([mask]*3,axis=-1)
        
        mask = mask*np.random.randint(0,256,(a,b,3)) + (1-mask)*img
        # mask = (mask - mask.min())/(mask.max()-mask.min())
        # mask = Image.fromarray((mask*255).astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.resize((256,256))
        mask.save(f'rand{limit}/{stem}_{k}.png')
        
        labels[f'{stem}_{k}.png'] = classname
    return labels

#Random Infilling
def random_infilling():
    
    Path(f'rand32').mkdir(exist_ok=True,parents=True)    
    
    for i in tqdm(data.getImgIds()):
        random_infilling_helper(32,i)


def mean_infilling_helper(limit,index):
    curr_anns = data.getAnnIds(imgIds=index)
    if len(curr_anns)<2:
        return
    
    stem = data.loadImgs(index)[0]['file_name'].split('.')[0]
    img = Image.open(f'PartImageNet/images/test/{data.loadImgs(index)[0]["file_name"]}')
    img = np.array(img,dtype=np.uint8)
    if img.ndim < 3:
        return
    img = i_t(img)
    img = rearrange(img, 'c h w -> h w c')
    for k,j in enumerate(getcombinations(curr_anns,limit)):
        try:
            mask = multiload(data,j)
        except:
            continue

        mask = np.where(mask>0,1,0)
        a,b = mask.shape
        mask = np.stack([mask]*3,axis=-1)
        M = np.array([0.485, 0.456, 0.406])
        M = np.tile(M,(a,b,1))  

        
        
        mask = (mask*M) + (1-mask)*img.numpy()
        # mask = (mask - mask.min())/(mask.max()-mask.min())
        # mask = Image.fromarray((mask*255).astype(np.uint8))
        mask = Image.fromarray((mask*255).astype(np.uint8))
        mask = mask.resize((256,256))  
        mask.save(f'mean{limit}/{stem}_{k}.png')
        

def mean_infilling():
    Path(f'mean32').mkdir(exist_ok=True,parents=True)   
    for i in tqdm(data.getImgIds()):
        mean_infilling_helper(32,i)



def get_metadata(limit):
    thala = {}
    for i in tqdm(data.getImgIds()):
        curr_anns = data.getAnnIds(imgIds=i)
        if len(curr_anns) < 2:
            continue
        
        stem = data.loadImgs(i)[0]['file_name'].split('.')[0]
        for k,j in enumerate(getcombinations(curr_anns,limit)):
            try:
                multiload(data,j)
            except:
                continue
            thala[f'{stem}_{k}.png'] = j
    with open(f'partset{limit}/metadata.json','w') as f:
        json.dump(thala,f)
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--limit', type=int,default=32)
    args = parser.parse_args()
    if args.method == 'rand32':
        random_infilling()
    elif args.method == 'mean32':
        mean_infilling()
    elif args.method == 'inpainting':
        generate_dataset(args)
    else:
        raise ValueError('Invalid method')
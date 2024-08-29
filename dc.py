from dataset import PartImageNetDataset2
from torch.utils.data import DataLoader
from evaluation_protocols import deletion_check_protocol
from explainer_wrapper import *
from tqdm import tqdm
from pathlib import Path
from utils import import_acc_to_model
import sys
import json
sys.path.append('models/ViT')

modelname = 'base' #Supported are base, dino, mae and large

import_acc_to_model(modelname)

exps = {
    'ViTBeyondIntuitionHExplainer':ViTBeyondIntuitionHExplainer,
    'ViTBeyondIntuitionTExplainer':ViTBeyondIntuitionTExplainer,
    'ViTGAExplainer':ViTGAExplainer,
    'ViTCAMExplainer':ViTCAMExplainer,
    'ViTRAMExplainer':ViTRAMExplainer,
    'ViTRolloutExplainer':ViTRolloutExplainer,
    'ViTPLRPExplainer':ViTPLRPExplainer,
    'ViTCheferLRPExplainer':ViTCheferLRPExplainer
}

#Note that TAM is proposed only for base model
if modelname in ['base', 'large']:
    exps['ViTTAMExplainer'] = ViTTAMExplainer

#Mention device here
dev = 'cuda:0'

#fullout means all parts are removed and then parts are added back
#mask means pixel masking
#inpainted means part-wise inpainting
dataset_type = 'mask' #Supported are mask, inpainted and fullout
if dataset_type not in ['mask','inpainted','fullout']:
    raise ValueError('Dataset type not supported')
dataset = PartImageNetDataset2('../PartImageNet/annotations/test/test.json',dataset_type,device=dev)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

D = {}

for key in tqdm(exps.keys()):
    if key != 'ViTCheferLRPExplainer' and key != 'ViTPLRPExplainer':
        model = vit_LRP(pretrained=True).to(torch.device(dev))
    else:
        model = CheferViT(pretrained=True).to(torch.device(dev))

    model.device = torch.device(dev)
    explainer = exps[key](model, dataset, dev, mae=True)
    dc = deletion_check_protocol(model, explainer, dataset, dataloader)
    
    D[key] = dc

#Dump D into json so that it is easy to load it later and access values
Path('Results/DC').mkdir(parents=True, exist_ok=True)
with open(f'Results/DC/{dataset_type}_{model}_dc.json', 'w') as f:
    json.dump(D, f)
    
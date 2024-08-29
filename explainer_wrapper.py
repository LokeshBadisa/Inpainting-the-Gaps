import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
from abc import abstractmethod
from PIL import Image

from models.ViT.ViT_explanation_generator import Baselines, LRP


class AbstractExplainer():
    def __init__(self, explainer, data, baseline = None):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.data = data
        self.explainer = explainer
        self.explainer_name = type(self.explainer).__name__
        self.baseline = baseline
        print(self.explainer_name)

    @abstractmethod
    def explain(self, input):
        return self.explainer.explain(self.model, input)
    
    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)

class AbstractAttributionExplainer(AbstractExplainer):
    
    @abstractmethod
    def explain(self, input):
        return self.explainer.explain(self.model, input)

    def get_important_parts(self, sample, target = None, thresholds= None):

        if thresholds is None:
            thresholds = self.get_p_thresholds()

        image = sample['image'].to(self.device)
        index = sample['index'].item()


        attribution = self.explain(image, target=target)
        attr_sum = attribution.sum()
                

        part_importances = self.get_part_importance(attribution, index)
        

        important_parts_for_thresholds = []
        for threshold in thresholds:
            important_parts = []            
            for key in part_importances.keys():
                if part_importances[key] > (attr_sum * threshold):
                    important_parts.append(key)
            important_parts_for_thresholds.append(important_parts)
        return important_parts_for_thresholds



    def get_part_importance(self, attribution, index):
        part_importances = {}

        dilation1 = nn.MaxPool2d(5, stride=1, padding=2)
        

        anns = self.data.loadAnns(self.data.getAnnIds(imgIds=index))

        for ann in anns:
            # pycocotools considers segmentation map with 4 corners as BBox rather than
            # segmentation which returns list rather than np.array. To counter this error,
            # we use if block here
            if len(ann['segmentation'][0]) <= 4:
                anns.remove(ann)
        

        for ann in anns:    
            mask = self.data.annToMask(ann)
            
            mask = torch.tensor(mask).float()
                
            mask = nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(224,224))
            mask = dilation1(mask)  
            
            part_importances[ann['id']] = (attribution * mask.to(self.device)).sum()
            
        

        return part_importances
    

    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)
    
class CaptumAttributionExplainer(AbstractAttributionExplainer):
    
    """
    A wrapper for Captum attribution methods.
    Args:
        explainer: Captum explanation method
    """
    def explain(self, input, target=None, baseline=None):
        if self.explainer_name == 'InputXGradient': 
            return self.explainer.attribute(input, target=target)
        elif self.explainer_name == 'IntegratedGradients':
            return self.explainer.attribute(input, target=target, baselines=self.baseline, n_steps=50)

# str_to_method = {
#                 'GradCam' : generate_cam_attn,
#                 # 'Rollout' : generate_rollout,
#                 'BIT' : generate_BIT,
#                 'BIH' : generate_BIH,
#                 'TA' : generate_transition_attention_maps,
#                 'GA' : generate_genattr,
#                 'CAM' : generate_cam_attn,
#                 'RAM' : generate_attn,
#                 # 'LRP' : generate_LRP,
#                 'TAM' : generate_transition_attention_maps,
#                 }

class ViTMethodExplainer(AbstractAttributionExplainer):
    def __init__(self, model, method):
        self.model = model
        self.explainer = Baselines(self.model)
        self.method = method

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.str_to_method[self.method](input, index=target).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.BILINEAR)
        attribution = m(attribution)
        return attribution

class ViTGradCamExplainer(AbstractAttributionExplainer):
    def __init__(self, model, data, device='cuda:2'):
        self.model = model
        self.data = data.data
        self.device = torch.device(device)
        self.explainer = Baselines(self.model)

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_cam_attn(input_, index=target).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.BILINEAR)
        attribution = m(attribution)
        return attribution
    
class ViTRolloutExplainer(AbstractAttributionExplainer):
    def __init__(self, model, data, device='cuda:2', mae=False, dino=False):
        self.model = model
        self.data = data.data
        self.device = torch.device(device)
        self.explainer = Baselines(self.model)
        self.mae = mae
        self.dino = dino

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_rollout(input_, start_layer=1,mae=self.mae).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.BILINEAR)
        attribution = m(attribution)
        return attribution

class ViTCheferLRPExplainer(AbstractAttributionExplainer):
    def __init__(self, model, data, device='cuda:2', mae=False, dino=False):
        self.model = model
        self.data = data.data
        self.device = torch.device(device)
        self.explainer = LRP(self.model)
        self.mae = mae
        self.dino = dino

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_LRP(input_, index=target, start_layer=1, method="transformer_attribution").reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.BILINEAR)
        attribution = m(attribution)
        return attribution
    
    
class ViTBeyondIntuitionTExplainer(AbstractAttributionExplainer):
    def __init__(self, model, data, device='cuda:2', mae=False, dino=False):
        self.model = model
        self.data = data.data
        self.device = torch.device(device)
        self.explainer = Baselines(self.model)
        self.mae = mae
        self.dino = dino

    def explain(self, input, target=None):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_BIT(input_, index=target, mae=self.mae, dino=self.dino).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.BILINEAR)
        attribution = m(attribution)
        return attribution
    
class ViTBeyondIntuitionHExplainer(AbstractAttributionExplainer):
    def __init__(self, model, data, device='cuda:2', mae=False, dino=False):
        self.model = model
        self.data = data.data
        self.device = torch.device(device)
        self.explainer = Baselines(self.model)
        self.mae = mae
        self.dino = dino

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_BIH(input_, index=target, mae=self.mae, dino=self.dino).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.BILINEAR)
        attribution = m(attribution)
        return attribution

# class ViTLBExplainer(AbstractAttributionExplainer):
#     def __init__(self, model):
#         self.model = model
#         self.explainer = Baselines(self.model)

#     def explain(self, input, target):
#         B,C,H,W = input.shape
#         assert B == 1
#         input_ = torch.nn.functional.interpolate(input, (224,224))
#         attribution = self.explainer.generate_gradsamplusplusv1wrong(input_, index=target).reshape(1, 1, 14, 14)
#         m = transforms.Resize((H,W), interpolation=Image.BILINEAR)
#         attribution = m(attribution)
#         return attribution
    
class ViTTAMExplainer(AbstractAttributionExplainer):
    def __init__(self, model, data, device='cuda:2', mae=False, dino=False):
        self.model = model
        self.data = data.data
        self.device = torch.device(device)
        self.explainer = Baselines(self.model)
        self.mae = mae
        self.dino = dino


    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_transition_attention_maps(input_, index=target,mae=self.mae,dino=self.dino).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.BILINEAR)
        attribution = m(attribution)
        return attribution

class ViTGAExplainer(AbstractAttributionExplainer):
    def __init__(self, model, data, device='cuda:2', mae=False, dino=False):
        self.model = model
        self.data = data.data
        self.device = torch.device(device)
        self.explainer = Baselines(self.model)
        self.mae = mae
        self.dino = dino

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_genattr(input_, index=target, mae=self.mae).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.BILINEAR)
        attribution = m(attribution)
        return attribution
    
class ViTCAMExplainer(AbstractAttributionExplainer):
    def __init__(self, model, data, device='cuda:2', mae=False, dino=False):
        self.model = model
        self.data = data.data
        self.device = torch.device(device)
        self.explainer = Baselines(self.model)
        self.mae = mae
        self.dino = dino

    def explain(self, input, target, mae=False, dino=False):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_cam_attn(input_, index=target, mae=self.mae).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.BILINEAR)
        attribution = m(attribution)
        return attribution
    
class ViTRAMExplainer(AbstractAttributionExplainer):
    def __init__(self, model, data, device='cuda:2', mae=False, dino=False):
        self.model = model
        self.data = data.data
        self.device = torch.device(device)
        self.explainer = Baselines(self.model)
        self.mae = mae
        self.dino = dino

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_attn(input_,mae=self.mae).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.BILINEAR)
        attribution = m(attribution)
        return attribution
    
class ViTPLRPExplainer(AbstractAttributionExplainer):
    def __init__(self, model, data, device='cuda:2', mae=False, dino=False):
        self.model = model
        self.data = data.data
        self.device = torch.device(device)
        self.explainer = LRP(self.model)
        self.mae = mae
        self.dino = dino

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_LRP(input_, index=target, start_layer=1, method="last_layer").reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.BILINEAR)
        attribution = m(attribution)
        return attribution

# class ViTMultExplainer(AbstractAttributionExplainer):
#     def __init__(self, model):
#         self.model = model
#         self.explainer = Baselines(self.model)

#     def explain(self, input, target):
#         B,C,H,W = input.shape
#         assert B == 1
#         input_ = torch.nn.functional.interpolate(input, (224,224))
#         attribution = self.explainer.generate_mult(input_).reshape(1, 1, 14, 14)
#         m = transforms.Resize((H,W), interpolation=Image.BILINEAR)
#         attribution = m(attribution)
#         return attribution

class CustomExplainer(AbstractExplainer):

    def explain(self, input):
        return 0
    
    def get_important_parts(self, image, part_map, target, colors_to_part, thresholds, with_bg = False):
        return 0
    
    def get_part_importance(self, image, part_map, target, colors_to_part, with_bg = False):
        return 0

    # if not inheriting from AbstractExplainer you need to add this function to your class as well
    #def get_p_thresholds(self):
    #    return np.linspace(0.01, 0.50, num=80)






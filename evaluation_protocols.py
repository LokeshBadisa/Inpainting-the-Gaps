from torch import no_grad, max, stack
from tqdm import tqdm
from utils import round_down
import numpy as np
from scipy.stats import spearmanr   
from collections import defaultdict


def accuracy(model, dataloader):
    model.eval()
    with no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(dataloader):
            outputs = model(images)
            _, predicted = max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def single_deletion_protocol(model, explainer, dataset, dataloader):

    correlations = []
    for sample in tqdm(dataloader):

        image = sample['image'].to(dataset.device)
        target = sample['class']
        idx = sample['index'].item()

        score = {}

        output = model(image)   
        original_score = output[0,target].item()

        # try:
        interventions = dataset.get_interventions(idx)
        for id in interventions.keys():
            output = model(interventions[id].to(dataset.device))
            score[id] = output[0,target].item() 
        # This error arises because dataloader has images which weren't generated while inpainting
        # i.e. those images don't have intervention images
        # So to counter this, we use except block here
        # except:
            # exceptionfile.write(str(idx) + '\n')
            # continue

        attribution = explainer.explain(image, target)
        
        part_importances = explainer.get_part_importance(attribution, idx)

        score_diffs = {}
        for score_key in score.keys():
            score_diffs[score_key] = original_score - score[score_key]


        score_diffs_normalized = []
        part_importances_normalized = []
        for key in score_diffs.keys():
            score_diffs_normalized.append(score_diffs[key]) # not necessary to normalize with spearmanr coefficient
            part_importances_normalized.append(part_importances[key].item()) # not necessary to normalize with spearmanr coefficient

        
        correlation, p_value = spearmanr(score_diffs_normalized, part_importances_normalized)
        
        
        import math
        if math.isnan(correlation):
            # print("Alert: NaN correlation detected")
            # totnans.append(idx)
            continue
        
        correlations.append(correlation * 0.5 + 0.5)


    # exceptionfile.close()
    try:
        return sum(correlations)/len(correlations)#, totnans
    except:
        return 'djtillu'


def preservation_check_protocol(model, explainer, dataset, dataloader):
    

    thresholds = explainer.get_p_thresholds()
    scores_for_thresholds = {}
    for threshold in thresholds:
        scores_for_thresholds[threshold] = []

    for sample in tqdm(dataloader):

        try:

            image = sample['image'].to(dataset.device)
            idx = sample['index'].item()


            with no_grad():
                model_prediction_original = model(image).argmax(1).item()        

            
            important_parts_for_thresholds = explainer.get_important_parts(sample,model_prediction_original)
            all_parts = set(dataset.data.getAnnIds(imgIds=idx))

            unimportant_parts = []
            for i in important_parts_for_thresholds:
                unimportant_parts.append(list(all_parts-set(i)))

                
            intervened_images = dataset.get_interventions2(idx, unimportant_parts)
            
            
            sorted_keys = sorted(intervened_images.keys())

            # Reason same as what I wrote in get_interventions2's before stacking
            if len(sorted_keys) == 0:
                continue
            
            # Stack images for batch processing
            stacked_intervened_images = []
            for threshold in sorted_keys:
                stacked_intervened_images.append(intervened_images[threshold])

        
            stacked_intervened_images = stack(stacked_intervened_images).squeeze(1).to(dataset.device)

            # There are cases where intervened_images is a single image and not a batch
            # This is because the image was not generated while inpainting, sometimes
            # leaving us with a single image.
            # if stacked_intervened_images.dim() == 3:
            #     stacked_intervened_images = stacked_intervened_images.unsqueeze(0)
            
            with no_grad():
                intervened_images_outputs = model(stacked_intervened_images).argmax(1)


            for j,i in enumerate(sorted_keys):
                if model_prediction_original == intervened_images_outputs[j].item():
                    scores_for_thresholds[i].append(1.)
                else:
                    scores_for_thresholds[i].append(0.)
        except:
            continue

    for threshold in thresholds:
        if len(scores_for_thresholds[threshold]) != 0:
            scores_for_thresholds[threshold] = sum(scores_for_thresholds[threshold]) / len(scores_for_thresholds[threshold])
        else:
            scores_for_thresholds[threshold] = 0.
    
    # print('Preservation Check Score: ', scores_for_thresholds)
    return scores_for_thresholds


def deletion_check_protocol(model, explainer, dataset, dataloader):    
    
    thresholds = explainer.get_p_thresholds()
    scores_for_thresholds = {}
    for threshold in thresholds:
        scores_for_thresholds[threshold] = []


    for sample in tqdm(dataloader):


        image = sample['image'].to(dataset.device)
        idx = sample['index'].item()

        with no_grad():
            model_prediction_original = model(image).argmax(1).item() 
        
        
        important_parts_for_thresholds = explainer.get_important_parts(sample,model_prediction_original)  
        
        intervened_images = dataset.get_interventions2(idx, important_parts_for_thresholds)
        

        sorted_keys = sorted(intervened_images.keys())
        if len(sorted_keys) == 0:
            continue
        
        stacked_intervened_images = []
        for threshold in sorted_keys:
            stacked_intervened_images.append(intervened_images[threshold])

    
        stacked_intervened_images = stack(stacked_intervened_images).squeeze(1).to(dataset.device)
    

        with no_grad():
            intervened_images_outputs = model(stacked_intervened_images).argmax(1)
        
        
        for j,i in enumerate(sorted_keys):
            if model_prediction_original == intervened_images_outputs[j].item():
                scores_for_thresholds[i].append(0.)
            else:
                scores_for_thresholds[i].append(1.)
               

    for threshold in thresholds:
        if len(scores_for_thresholds[threshold]) != 0:
            scores_for_thresholds[threshold] = sum(scores_for_thresholds[threshold]) / len(scores_for_thresholds[threshold])
        else:
            scores_for_thresholds[threshold] = 0.

   
    # print('Deletion Check Scores: ', scores_for_thresholds)
    return scores_for_thresholds


def perturbation_test(model, explainer, dataset, dataloader, vis_class = 'top', mode = 'positive'):
    scores = defaultdict(list)

    for sample in tqdm(dataloader):

        image = sample['image'].to(dataset.device)        
        idx = sample['index'].item()

        with no_grad():
            model_output = model(image).argmax(1).item()

        target = model_output if vis_class == 'top' else sample['class']

        attribution = explainer.explain(image, target)
        
        
        part_importances = explainer.get_part_importance(attribution, idx)
        
        intervened_images, areas, total_area = dataset.get_incremental_interventions(idx, part_importances, mode)

        with no_grad():
            for image, area in zip(intervened_images.values(), areas):
                output = model(image.to(dataset.device)).argmax(1).item()
                if output == target:
                    scores[round_down(area/total_area)].append(1.)
                else:
                    scores[round_down(area/total_area)].append(0.)
            

    scores = dict(scores)   
    for i in scores:
        scores[i] = np.mean(scores[i])

    return scores

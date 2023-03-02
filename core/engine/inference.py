import torch
import logging
import numpy as np
from tqdm import tqdm
from core.engine import losses as mylosses

def eval_dataset(cfg, model, data_loader, device, model_type='pytorch', class_weights=None):
    logger = logging.getLogger("CORE.inference")

    # Create losses
    if class_weights is None:
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.tensor(class_weights, dtype=torch.float).to(cfg.MODEL.DEVICE))

    # criterion_jaccard = mylosses.JaccardIndex(reduction=False)
    criterion_dice = mylosses.DiceLoss(reduction=False)

    stats = {
        'sample_count': 0.0,
        'loss': 0.0,
        'jaccard': 0.0,
        'dice': 0.0
    }

    for data_entry in tqdm(data_loader):
        images, labels, masks = data_entry

        # Forward images
        with torch.no_grad():
            # B,C,H,W = images.shape
            if model_type == 'onnx':
                images_np = images.numpy().astype(np.float32)
                outputs = model.forward(images_np, preprocess=False, postprocess=False)
                outputs = torch.from_numpy(outputs).to(device)
            elif model_type == 'tensorrt':
                images_np = images.numpy().astype(np.float32)
                outputs = model.forward(images_np, preprocess=False, postprocess=False)
                outputs = torch.from_numpy(outputs).to(device) 
            elif model_type == 'pytorch':
                images = images.to(device)
                outputs = model(images)
            else:
                logger.error("Unknown model type: %s. Aborting...".format(model_type))
                return -1

        # Calculate loss
        targets = labels.to(device)
        masks = masks.to(device)

        losses = cross_entropy_loss.forward(outputs, targets.type(torch.long))

        # Apply mask
        losses = losses * masks

        # Calculate metrics
        outputs_classified = torch.softmax(outputs, dim=1).argmax(dim=1)

        outputs_classes, target_classes = [], []
        for class_id in range(outputs.shape[1]):
            if class_id == 0: # Skip background
                continue
            outputs_classes.append(torch.where(outputs_classified == class_id, float(class_id), 0.0))
            target_classes.append(torch.where(targets == class_id, float(class_id), 0.0))

        # jaccard_losses = criterion_jaccard.forward(torch.stack(outputs_classes, dim=1), torch.stack(target_classes, dim=1), masks)
        dice_losses = criterion_dice.forward(torch.stack(outputs_classes, dim=1), torch.stack(target_classes, dim=1), masks)

        # Reduce loss (mean)
        stats['loss'] += torch.mean(losses).item()
        stats['jaccard'] += 0.0 #torch.mean(jaccard_losses).item()
        stats['dice'] += torch.mean(dice_losses).item()
        stats['sample_count'] += 1

    # Return results
    stats['loss'] /= stats['sample_count']
    stats['jaccard'] /= stats['sample_count']
    stats['dice'] /= stats['sample_count']

    result_dict = {
        'loss': stats['loss'],
        'jaccard': stats['jaccard'],
        'dice': stats['dice']
    }

    return result_dict


def eval_dataset_ships(cfg, model, data_loader, device, model_type='pytorch'):
    logger = logging.getLogger("CORE.inference")

    stats = {
        'sample_count': 0.0,
        'loss_sum': 0.0,
        'loss_classifier': 0.0,
        'loss_box_reg': 0.0,
        'loss_objectness': 0.0,
        'loss_rpn_box_reg': 0.0
    }

    for data_entry in tqdm(data_loader):
        images, rects, masks, rects_real_num = data_entry

        # Forward images
        with torch.no_grad():
            if model_type == 'pytorch':
                images = list(image.to(device) for image in images)

                targets = []
                for i in range(len(images)):
                    d = {}
                    if rects_real_num[i] == 0:
                        d['boxes'] = torch.empty((0, 4), dtype=torch.float32).to(device)
                        d['labels'] = torch.empty((0,), dtype=torch.int64).to(device)
                    else:
                        real_rects = rects[i][0:rects_real_num[i]]
                        d['boxes'] = real_rects.to(device)
                        d['labels'] = torch.ones(len(real_rects), dtype=torch.int64).to(device)
                    targets.append(d)

                loss_dict = model.model(images, targets)
            else:
                logger.error("Unknown model type: %s. Aborting...".format(model_type))
                return -1

        # Calculate loss
        loss = sum(loss for loss in loss_dict.values())
        stats['loss_sum'] += loss.item()
        stats['loss_classifier'] += loss_dict["loss_classifier"].item()
        stats['loss_box_reg'] += loss_dict["loss_box_reg"].item()
        stats['loss_objectness'] += loss_dict["loss_objectness"].item()
        stats['loss_rpn_box_reg'] += loss_dict["loss_rpn_box_reg"].item()
        stats['sample_count'] += 1

    # Return results
    stats['loss_sum'] /= stats['sample_count']
    stats['loss_classifier'] /= stats['sample_count']
    stats['loss_box_reg'] /= stats['sample_count']
    stats['loss_objectness'] /= stats['sample_count']
    stats['loss_rpn_box_reg'] /= stats['sample_count']

    result_dict = {
        'loss_sum': stats['loss_sum'],
        'loss_classifier': stats['loss_classifier'],
        'loss_box_reg': stats['loss_box_reg'],
        'loss_objectness': stats['loss_objectness'],
        'loss_rpn_box_reg': stats['loss_rpn_box_reg'],
    }

    return result_dict
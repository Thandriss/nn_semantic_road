import os
import time
import torch
import logging
import numpy as np
import cv2 as cv
from tqdm import tqdm
from core.utils import dist_util
from torch.utils.tensorboard import SummaryWriter
from .inference import eval_dataset
from core.data import make_data_loader
from core.data.transforms.transforms2 import UnStandardize, ToCV2Image, ToTensor
from core.engine import losses as mylosses
from torchvision.utils import make_grid

def do_eval(cfg, model, class_weights, distributed, **kwargs):
    torch.cuda.empty_cache()

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    data_loader = make_data_loader(cfg, False)

    model.eval()

    device = torch.device(cfg.MODEL.DEVICE)
    # torch.cuda.set_device(device);

    result_dict = eval_dataset(cfg, model, data_loader, device, 'pytorch', class_weights)

    torch.cuda.empty_cache()
    return result_dict


def draw_masks(image, classes_mask):
    color_mask = np.zeros_like(image, dtype=np.uint8) # TODO: 3-channel
    # color_mask[np.squeeze(classes_mask == 1, -1), :] = (255, 128, 0) # river
    # color_mask[np.squeeze(classes_mask == 2, -1), :] = (255, 222, 0) # water
    color_mask[classes_mask == 1, :] = (0, 128, 255) #(255, 128, 0) # river
    color_mask[classes_mask == 2, :] = (239, 255, 0) #(0, 255, 239) # water
    return cv.addWeighted(image, 0.85, color_mask, 0.15, 1.0)


def do_train(cfg,
             model,
             data_loader,
             class_weights,
             device,
             arguments,
	     args):
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)


    logger = logging.getLogger("CORE")
    logger.info("Start training ...")

    # Set model to train mode
    model.train()

    # Create tensorboard writer
    save_to_disk = dist_util.is_main_process()
    if args.use_tensorboard and save_to_disk:
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    # Prepare to train
    iters_per_epoch = len(data_loader)
    total_steps = iters_per_epoch * cfg.SOLVER.MAX_ITER
    start_epoch = arguments["epoch"]
    logger.info("Iterations per epoch: {0}. Total steps: {1}. Start epoch: {2}".format(iters_per_epoch, total_steps, start_epoch))

    # Create losses
    if class_weights is None:
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.tensor(class_weights, dtype=torch.float, device=cfg.MODEL.DEVICE))

    criterion_jaccard = mylosses.JaccardIndex(reduction=False)
    criterion_dice = mylosses.DiceLoss(reduction=False)

    # Create additional transforms
    unstandardize = UnStandardize()
    toCVimage = ToCV2Image()
    totensor = ToTensor()

    # Epoch loop
    for epoch in range(start_epoch, cfg.SOLVER.MAX_ITER):
        arguments["epoch"] = epoch + 1

        # Create progress bar
        print(('\n' + '%10s' * 6) % ('Epoch', 'gpu_mem', 'lr', 'loss', 'jaccard', 'dice'))
        pbar = enumerate(data_loader)
        pbar = tqdm(pbar, total=len(data_loader))

        # Prepare data for tensorboard
        best_samples, worst_samples = [], []

        # Iteration loop
        loss_sum, jaccard_loss_sum, dice_loss_sum = 0.0, 0.0, 0.0

        for iteration, data_entry in pbar:
            global_step = epoch * iters_per_epoch + iteration

            images, labels, masks = data_entry

            # Forward data to GPU
            # images = images.to(device)
            images = images.to(device)
            targets = labels.to(device)
            masks = masks.to(device)

            # Do prediction
            outputs = model(images)



            # Calculate loss
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
            ################### Best images
            if cfg.TENSORBOARD.BEST_SAMPLES_NUM > 0:
                
                with torch.no_grad():
                    losses_ = dice_losses.detach().clone()
                    # Select only non-empty labels
                    cnz = torch.count_nonzero(labels, dim=(1,2))
                    idxs = torch.tensor([i for i,v in enumerate(cnz) if v > 0])
                

                    if not torch.numel(idxs):
                        continue
                    # Find best metric
                    losses_selected = torch.index_select(losses_.to('cpu'), 0, idxs)
                    # losses_selected = torch.index_select(losses_.to('cpu'), 0, idxs)
                    losses_per_image = torch.mean(losses_selected, dim=(1))
                    max_idx = torch.argmax(losses_per_image).item()
                    best_loss = losses_per_image[max_idx].item()

                    # Check if better than existing
                    need_save, id_to_remove = True, -1
                    if len(best_samples) >= cfg.TENSORBOARD.BEST_SAMPLES_NUM:
                        id_to_remove = min(range(len(best_samples)), key=lambda x : best_samples[x][0])
                        if best_loss < best_samples[id_to_remove][0]:
                            need_save = False

                    if need_save:
                        # Prepare tensorboard image
                        labels_selected = torch.index_select(labels, 0, idxs)
                        best_label = labels_selected[max_idx, :, :]
                        best_label = torch.stack([best_label, best_label, best_label], dim=0) # to 3-channel image

                        outputs_selected = torch.index_select(outputs.detach().to('cpu'), 0, idxs)
                        # outputs_selected = torch.index_select(outputs.detach().to('cpu'), 0, idxs)
                        best_output = outputs_selected[max_idx, :, :]
                        best_output = best_output.softmax(dim=0).argmax(dim=0)
                        # best_output = torch.stack([best_output, best_output, best_output], dim=0) # to 3-channel image

                        images_selected = torch.index_select(images.detach().to('cpu'), 0, idxs)
                        best_image = images_selected[max_idx]
                        best_image = toCVimage(best_image).copy()
                        # best_image, _, _ = unstandardize(best_image)
                        best_image = (255.0 * best_image).astype(np.uint8)

                        best_image = draw_masks(best_image, best_output)

                        # save_img = torch.cat([best_image, best_label, best_output], dim=2)
                        best_image = best_image.astype(np.float32) / 255.0
                        save_img, _, _ = totensor(best_image)

                        # Save image
                        if id_to_remove != -1:
                            del best_samples[id_to_remove]

                        best_samples.append((best_loss, save_img))
            ###############################

            ################### Worst images
            if cfg.TENSORBOARD.WORST_SAMPLES_NUM > 0:
                with torch.no_grad():
                    losses_ = dice_losses.detach().clone()

                    # Select only non-empty labels
                    cnz = torch.count_nonzero(labels, dim=(1,2))
                    idxs = torch.tensor([i for i,v in enumerate(cnz) if v > 0])
                    if not torch.numel(idxs):
                        continue

                    # Find worst metric
                    losses_selected = torch.index_select(losses_.to('cpu'), 0, idxs)
                    losses_per_image = torch.mean(losses_selected, dim=(1))
                    min_idx = torch.argmin(losses_per_image).item()
                    worst_loss = losses_per_image[min_idx].item()

                    # Check if worse than existing
                    need_save, id_to_remove = True, -1
                    if len(worst_samples) >= cfg.TENSORBOARD.WORST_SAMPLES_NUM:
                        id_to_remove = max(range(len(worst_samples)), key=lambda x : worst_samples[x][0])
                        if worst_loss > worst_samples[id_to_remove][0]:
                            need_save = False

                    if need_save:
                        # Prepare tensorboard image
                        labels_selected = torch.index_select(labels, 0, idxs)
                        worst_label = labels_selected[min_idx, :, :]
                        worst_label = torch.stack([worst_label, worst_label, worst_label], dim=0) # to 3-channel image

                        outputs_selected = torch.index_select(outputs.detach().to('cpu'), 0, idxs)
                        worst_output = outputs_selected[min_idx, :, :]
                        worst_output = worst_output.softmax(dim=0).argmax(dim=0)
                        # worst_output = torch.stack([worst_output, worst_output, worst_output], dim=0) # to 3-channel image

                        images_selected = torch.index_select(images.detach().to('cpu'), 0, idxs)
                        worst_image = images_selected[min_idx]
                        worst_image = toCVimage(worst_image).copy()
                        # worst_image, _, _ = unstandardize(worst_image)
                        worst_image = (255.0 * worst_image).astype(np.uint8)

                        worst_image = draw_masks(worst_image, worst_output)
                        # save_img = torch.cat([worst_image, worst_label, worst_output], dim=2)

                        worst_image = worst_image.astype(np.float32) / 255.0
                        save_img, _, _ = totensor(worst_image)

                        # Save image
                        if id_to_remove != -1:
                            del worst_samples[id_to_remove]

                        worst_samples.append((worst_loss, save_img))
            ###############################

            # Reduce loss (mean)
            loss = torch.mean(losses)
            loss_sum += loss.item()
            # jaccard_loss_sum += torch.mean(jaccard_losses).item()
            dice_loss_sum += torch.mean(dice_losses).item()

            # Do optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Update progress bar
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0) # (GB)
            s = ('%10s' * 2 + '%10.4g' * 4) % (
                                                '%g/%g' % (epoch, cfg.SOLVER.MAX_ITER - 1),
                                                mem,
                                                optimizer.param_groups[0]['lr'],
                                                loss_sum / (iteration + 1),
                                                jaccard_loss_sum / (iteration + 1),
                                                dice_loss_sum / (iteration + 1))

            pbar.set_description(s)

        # scheduler.step()

        # Do evaluation
        if args.eval_step > 0 and epoch % args.eval_step == 0:
            print('\nEvaluation ...')
            result_dict = do_eval(cfg, model, class_weights, distributed=args.distributed, iteration=global_step)
            print(('\n' + 'Evaluation results:' + '%10s' * 3) % ('loss', 'jaccard', 'dice'))
            print('                   ' + '%10.4g%10.4g%10.4g' % (result_dict['loss'], result_dict['jaccard'], result_dict['dice']))

            if summary_writer:
                summary_writer.add_scalar('losses/validation_loss', result_dict['loss'], global_step=global_step)
                summary_writer.add_scalar('metrics/validation_jaccard', result_dict['jaccard'], global_step=global_step)
                summary_writer.add_scalar('metrics/validation_dice', result_dict['dice'], global_step=global_step)
                summary_writer.flush()

            model.train()

        # Save epoch results
        if epoch % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(global_step), **arguments)

            if summary_writer:
                with torch.no_grad():
                    # Best samples
                    if len(best_samples):
                        tb_images = [sample[1] for sample in best_samples]
                        image_grid = torch.stack(tb_images, dim=0)
                        # image_grid = np.stack(tb_images, dim=0)
                        image_grid = make_grid(image_grid, nrow=1)
                        summary_writer.add_image('images/train_best_samples', image_grid, global_step=global_step)

                    # Worst samples
                    if len(worst_samples):
                        tb_images = [sample[1] for sample in worst_samples]
                        image_grid = torch.stack(tb_images, dim=0)
                        # image_grid = np.stack(tb_images, dim=0)
                        image_grid = make_grid(image_grid, nrow=1)
                        summary_writer.add_image('images/train_worst_samples', image_grid, global_step=global_step)

                    summary_writer.add_scalar('losses/loss', loss_sum / (iteration + 1), global_step=global_step)
                    # summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], global_step=global_step)
                    summary_writer.add_scalar('metrics/jaccard', jaccard_loss_sum / (iteration + 1), global_step=global_step)
                    summary_writer.add_scalar('metrics/dice', dice_loss_sum / (iteration + 1), global_step=global_step)
                    summary_writer.flush()

    # Save final model
    checkpointer.save("model_final", **arguments)

    return model
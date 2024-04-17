# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data.distributed
from tensorboardX import SummaryWriter 
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from tqdm import tqdm
from utils.utils import dice
from utils.metric import calculate_score
import nibabel as nib
from peft import set_peft_model_state_dict
from monai.inferers import sliding_window_inference
import random 
from rouge import Rouge 


def test_model(model, test_loader, args):
        
    pretrained_pth = os.path.join(args.pretrained_dir, args.pretrained_model_name)
    model_dict = torch.load(pretrained_pth, map_location='cpu')["state_dict"]
    model.load_state_dict(model_dict, strict=False)
    model.eval()
    model.to(args.gpu)

    eval_tag = ('_%s'%args.report_dir[-1].split('/')[-1].split('.')[0]) + args.ablation 
    file = open(args.pretrained_dir + '/result%s.csv'%eval_tag, 'w')
    file.write("\nno,ID,Avr,Dice,IoU,HD,Report")
    raw_dir = args.pretrained_dir + '/raw%s'%eval_tag
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    with torch.no_grad():

        dice_list_case = []
        dice_list_organ = []
        iou_list_sub = []
        hd_list_sub = []

        for j, batch in enumerate(test_loader):
            val_inputs, val_labels = (batch["image"].to(args.gpu), batch["label"].to(args.gpu))
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
            print("\nInference on case {}".format(img_name))
            
            with autocast(enabled=args.amp):
                if args.context:
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs, roi_size=(args.roi_x, args.roi_y, args.roi_z), sw_batch_size=args.sw_batch_size, predictor=model, overlap=args.infer_overlap, mode="gaussian", report_in=batch["report"],
                    )
                else:
                    val_outputs = sliding_window_inference(
                        inputs=val_inputs, roi_size=(args.roi_x, args.roi_y, args.roi_z), sw_batch_size=args.sw_batch_size, predictor=model, overlap=args.infer_overlap, mode="gaussian"
                    )

            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]

            dice_list_sub = []
            for i in range(1, args.c_max):
                organ_Dice = dice(val_outputs == i, val_labels == i)
                dice_list_sub.append(organ_Dice)
                if i == 1:
                    iou_PTV = calculate_score(val_outputs == i, val_labels == i, "iou")
                    iou_list_sub.append(iou_PTV)
                    hd_PTV = calculate_score(val_outputs == i, val_labels == i, "hd")
                    hd_list_sub.append(hd_PTV)
            mean_dice = np.mean(dice_list_sub)

            file.write("\n%d,%s,%.3f,"%(j,img_name,mean_dice))
            [file.write("%.3f,%.3f,%.3f"%(dice_o, iou_PTV, hd_PTV)) for dice_o in dice_list_sub]
            dice_list_case.append(mean_dice)
            dice_list_organ.append(np.array(dice_list_sub))

            # save image
            if (j % args.save_interval == 0):
                nib.save(
                    nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(raw_dir, img_name))
                nib.save(
                    nib.Nifti1Image((batch["image"][0][0]*255).astype(np.uint8), original_affine), os.path.join(raw_dir, img_name + '_data'))
                nib.save(
                    nib.Nifti1Image(batch["label"][0][0].astype(np.uint8), original_affine), os.path.join(raw_dir, img_name + '_label'))


        iou_list_sub = np.array(iou_list_sub)
        hd_list_sub = np.array(hd_list_sub)
        dice_list_organ = np.swapaxes(np.array(dice_list_organ), 0, 1)

        file.write("\nAvr,%.3f,%.3f,%.3f,"%(np.mean(dice_list_case), np.mean(iou_list_sub), np.mean(hd_list_sub)))
        file.write("\nStd,%.3f,%.3f,%.3f,"%(np.std(dice_list_case), np.std(iou_list_sub), np.std(hd_list_sub)))
        file.close()
        
    return np.mean(dice_list_case)


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    run_loss = AverageMeter()
    
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.gpu), target.cuda(args.gpu)
        for param in model.parameters():
            param.grad = None
            
        with autocast(enabled=args.amp):
            
            if args.context:
                logits = model(data, report_in = batch_data["report"], target=target)
            else:
                logits = model(data)
                
            loss = loss_func(logits, target)
        
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        
    for param in model.parameters():
        param.grad = None
        
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None, loss_func=None):
    model.eval()
    dice_list_case = []
    run_loss = AverageMeter()
    
    if args.test_mode:
        raw_dir = args.logdir  + '/raw' + ('_ext1' if args.test_mode == 2 else '')
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
    

    with torch.no_grad():
        
        for idx, batch_data in enumerate(loader):
            dice_list_sub = []
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    if args.context:
                        logits = model_inferer(data, report_in = batch_data["report"])
                    else:
                        logits = model_inferer(data)
                else:
                    if args.context:
                        logits = model(data, report_in = batch_data["report"])
                    else:
                        logits = model(data)
                    
            loss = loss_func(logits, target)

            if args.distributed:
                loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                run_loss.update(
                    np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
                )
            else:
                run_loss.update(loss.item(), n=args.batch_size)
            
            logits = F.interpolate(logits, size=target.shape[-3:], mode='nearest')
            val_outputs = torch.softmax(logits, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_labels = target.cpu().numpy()[0, 0, :, :, :]
            
            for i in range(1, args.c_max):
                organ_Dice = dice(val_outputs == i, val_labels == i)
                dice_list_sub.append(organ_Dice)
            dice_list_case.append(dice_list_sub)
    
    return dice_list_case, run_loss.avg


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
        log_name = os.path.join(args.logdir, 'loss_log.txt')
        with open(log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('\n\n%s\nepoch, loss, mean, ptv, ctv, gtv, organs\n'% now)

    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    val_acc_max_epoch = 0
    
    pbar = tqdm(range(start_epoch, args.max_epochs))
    
    for _, epoch in enumerate(pbar):
        
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
            
        # train
        print("epoch >> %3d"%(epoch))
        if not args.test_mode:
            train_loss = train_epoch(
                model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
            )
            pbar.set_postfix(loss=train_loss)

            if args.rank == 0 and writer is not None:
                writer.add_scalar("train_loss", train_loss, epoch)

        # val
        b_new_best = False
        if (epoch % args.val_every == 0): 
            if args.distributed:
                torch.distributed.barrier()

            val_acc, val_loss = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
                loss_func=loss_func,
            )
            
            pbar.set_postfix(val_loss=val_loss)
            
            if args.test_mode:
                print("Test Finished !, Best Accuracy: ", val_acc)
                return val_acc

            val_acc = np.swapaxes(np.array(val_acc), 0, 1)
            val_acc_list = np.mean(val_acc, axis=1)
            val_avg_acc = np.mean(val_acc_list)

            message = '%d, %.3f, %.6f, '%(epoch, val_loss, val_avg_acc) + ', '.join(['%.3f'%acc for acc in val_acc_list])

            if args.logdir is not None and args.rank == 0:
                with open(log_name, "a") as log_file:
                    log_file.write('%s\n' % message)  # save the message
                
            if args.rank == 0:
                
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                
                if (epoch > 0) & (val_avg_acc > val_acc_max):
                    print("new best ({:.6f} --> {:.6f} @ {:d} epoch). ".format(val_acc_max, val_avg_acc, epoch))
                    val_acc_max = val_avg_acc
                    val_acc_max_epoch = epoch
                    b_new_best = True
                else:
                    print("current ({:.6f}), best ({:.6f} @ {:d} epoch). ".format(val_avg_acc, val_acc_max, val_acc_max_epoch))
                        
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:

                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_last.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_last.pt"), os.path.join(args.logdir, "model_best.pt"))
        
        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)
    
    if args.logdir is not None and args.rank == 0:
        with open(log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('\nFinished at %s'% now)

    return val_acc_max

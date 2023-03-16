import warnings

warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from utils.hparams import HParam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import dataloader
from utils import metrics
from core.resunetplusplus import ResUnetPlusPlus
from utils.logger import MyWriter, create_logger
import torch
import argparse
import os
import time
import datetime
from timm.utils import AverageMeter


def main(hp, num_epochs, resume, name):

    # checkpoint_dir = "{}/{}".format(hp.checkpoints, name)
    # os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs("{}/{}".format(hp.log, name), exist_ok=True)
    writer = MyWriter("{}/{}".format(hp.log, name))
    # get model

    model = ResUnetPlusPlus(3)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of param: {n_parameters}')
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    
    # set up binary cross entropy and dice loss
    criterion = metrics.BCEDiceLoss()

    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    optimizer = torch.optim.NAdam(model.parameters(), lr=hp.lr)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    

    # starting params
    best_loss = 999
    start_epoch = 0
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            logger.info("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            start_epoch = checkpoint["epoch"]

            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # get data
    train_dataset = dataloader.ImageDataset(hp.train, transform=transforms.Compose([dataloader.ToTensorTarget()]))
    val_dataset = dataloader.ImageDataset(hp.valid, transform=transforms.Compose([dataloader.ToTensorTarget()]))

    # creating loaders
    train_dataloader = DataLoader(train_dataset, batch_size=hp.batch_size, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)

    step = 0
    for epoch in range(start_epoch, num_epochs):        
        model.train()
        optimizer.zero_grad()

        num_steps = len(train_dataloader)
        batch_time = AverageMeter()

        # step the learning rate scheduler
        lr_scheduler.step()

        # run training and validation
        # logging accuracy and loss
        train_acc = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()
        # iterate over data

        start = time.time()
        end = time.time()
        for idx, data in enumerate(train_dataloader):
            # get the inputs and wrap in Variable
            inputs = data["image"].cuda()
            targets = data["mask"].cuda()

            # forward
            # prob_map = model(inputs) # last activation was a sigmoid
            # outputs = (prob_map > 0.3).float()
            outputs = model(inputs)
            # outputs = torch.nn.functional.sigmoid(outputs)

            loss = criterion(outputs, targets)

            # backward
            loss.backward()
            optimizer.step()

            train_acc.update(metrics.dice_coeff(outputs, targets), outputs.size(0))
            train_loss.update(loss.data.item(), outputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % hp.PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                logger.info(
                    f'Train: [{epoch}/{num_epochs}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'train_loss {train_loss.val:.4f} ({train_loss.avg:.4f})\t'
                    f'train_acc {train_acc.val:.4f} ({train_acc.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')

            # tensorboard logging
            if step % hp.logging_step == 0:
                writer.log_training(train_loss.avg, train_acc.avg, step)                
                            
            step += 1
        
        epoch_time = time.time() - start
        logger.info(f' * Train_Acc {train_acc.avg:.4f}')
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

        # Validation
        valid_metrics = validation(val_dataloader, model, criterion, writer, epoch)
        save_path = os.path.join(checkpoint_dir, "%s_checkpoint_%04d.pt" % (name, epoch))
        # store best loss and save a model checkpoint
        bets_loss = min(valid_metrics["valid_loss"], best_loss)
        torch.save(
            {
                "step": step,
                "epoch": epoch,
                "arch": "ResUNet",
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict()
            },
            save_path)
        logger.info(f'Saved checkpoint to: {save_path}')


def validation(valid_loader, model, criterion, tb_logger, step):

    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()

    # Iterate over data.
    end = time.time()
    for idx, data in enumerate(valid_loader):
        # get the inputs and wrap in Variable
        inputs = data["image"].cuda()
        targets = data["mask"].cuda()

        # forward
        # prob_map = model(inputs) # last activation was a sigmoid
        # outputs = (prob_map > 0.3).float()
        outputs = model(inputs)
        # outputs = torch.nn.functional.sigmoid(outputs)

        loss = criterion(outputs, targets)

        valid_acc.update(metrics.dice_coeff(outputs, targets), outputs.size(0))
        valid_loss.update(loss.data.item(), outputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end =  time.time()

        if idx % hp.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)            
            logger.info(
                f'Validation: [{idx}/{len(valid_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Val_loss {valid_loss.val:.4f} ({valid_loss.avg:.4f})\t'
                f'Val_acc {valid_acc.val:.4f} ({valid_acc.avg:.4f})\t'                
                f'Mem {memory_used:.0f}MB')    

        if idx == 0:
            tb_logger.log_images(inputs.cpu(), targets.cpu(), outputs.cpu(), step)

    logger.info(f' * Val_acc {valid_acc.avg:.4f}')
    tb_logger.log_validation(valid_loss.avg, valid_acc.avg, step)
    # model.train()
    return {"valid_loss": valid_loss.avg, "valid_acc": valid_acc.avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gas Extraction")
    parser.add_argument("-c", "--config", type=str, required=True, help="yaml file for configuration")
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--name", default="default", type=str, help="Experiment name")

    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())

    checkpoint_dir = "{}/{}".format(hp.checkpoints, args.name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = create_logger(output_dir=checkpoint_dir, name=hp.MODEL.NAME)

    main(hp, num_epochs=args.epochs, resume=args.resume, name=args.name)
import os
import sys
import time
import torch
import torchvision
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table, flop_count


import loader
import models
from models.common import config
from test import testing


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f"checking paths, mkdir: {path}")


def main():
    check_path(config.para.save_path)
    check_path(config.para.folder)
    set_seed(996007)

    net = models.AUV_Net(layer_num=7,resolution=config.para.patch_size,rate=config.para.rate).train().to(config.para.device)
    print("para num: ",sum(p.numel() for p in net.parameters() if p.requires_grad))
    tensor = torch.rand(1, 1, 256, 256)
    flops = FlopCountAnalysis(net, tensor.to(config.para.device))
    print(flop_count_table(flops))
    exit(0)
    optimizer = optim.AdamW(filter(lambda x: x.requires_grad, net.parameters()), lr=config.para.lr)
    # scheduler = LS.MultiStepLR(optimizer, milestones=[30, 100, 175], gamma=0.1)
    scheduler = LS.CosineAnnealingLR(optimizer,T_max=50,eta_min=1e-7)

    if os.path.exists(config.para.my_state_dict):
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(config.para.my_state_dict, map_location=config.para.device))
            info = torch.load(config.para.my_info, map_location=config.para.device)
        else:
            raise Exception(f"No GPU.")

        start_epoch = info["epoch"]
        current_best = info["res"]
        print(f"Loaded trained model of epoch {start_epoch}, res: {current_best}.")
    else:
        start_epoch = 1
        current_best = 0
        print("No saved model, start epoch = 1_back.")

    print("Data loading...")

    train_set = loader.TrainDatasetFromFolder('../dataset/BSD400', block_size=config.para.patch_size)
    dataset_train = torch.utils.data.DataLoader(
        dataset=train_set, num_workers=16, batch_size=config.para.batch_size, shuffle=True, pin_memory=True)
    # dataset_train = loader.train_loader()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    over_all_time = time.time()
    for epoch in range(start_epoch, int(200)):
        print("Please note:    Lr: {}.\n".format(optimizer.param_groups[0]['lr']))

        epoch_loss = 0.
        dic = {"epoch": epoch, "device": config.para.device, "rate": config.para.rate}

        for idx, xi in enumerate(tqdm(dataset_train, desc="Now training: ", postfix=dic)):
            with torch.cuda.amp.autocast(enabled=True):
                xi = xi.to(config.para.device)
                optimizer.zero_grad()
                xo = net(xi)
                batch_loss = torch.mean(torch.pow(xo - xi, 2)).to(config.para.device)
                epoch_loss += batch_loss.item()

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if idx % 10 == 0:
                tqdm.write("\r[{:5}/{:5}], Loss: [{:8.6f}]".format(
                    config.para.batch_size * (idx + 1),
                    dataset_train.__len__() * config.para.batch_size,
                    batch_loss.item()))

        avg_loss = epoch_loss / dataset_train.__len__()
        print("\n=> Epoch of {:2}, Epoch Loss: [{:8.6f}]".format(epoch, avg_loss))

        # Make a log note.
        if epoch == 1:
            if not os.path.isfile(config.para.my_log):
                output_file = open(config.para.my_log, 'w')
                output_file.write("=" * 120 + "\n")
                output_file.close()
            output_file = open(config.para.my_log, 'r+')
            old = output_file.read()
            output_file.seek(0)
            output_file.write("\nAbove is {} test. Note：{}.\n"
                              .format("???", None) + "=" * 120 + "\n")
            output_file.write(old)
            output_file.close()

        with torch.no_grad():
            p, s = testing(net.eval(), val=True, save_img=True)
        print("{:5.3f}".format(p))
        if p > current_best:
            epoch_info = {"epoch": epoch, "res": p}
            torch.save(net.state_dict(), config.para.my_state_dict)
            torch.save(epoch_info, config.para.my_info)
            print("Check point saved\n")
            current_best = p
            output_file = open(config.para.my_log, 'r+')
            old = output_file.read()
            output_file.seek(0)

            output_file.write(f"Epoch {epoch}, Loss of train {round(avg_loss, 6)}, Res {round(current_best, 2)}, {round(s, 4)}\n")
            output_file.write(old)
            output_file.close()
        scheduler.step()
        print("Epoch time: {:.3f}s".format(time.time() - over_all_time))

    print("Train end.")
    

def gpu_info():
    memory = int(os.popen('nvidia-smi | grep %').read()
                 .split('C')[int(config.para.device.split(':')[1]) + 1].split('|')[1].split('/')[0].split('MiB')[0].strip())
    return memory


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.memory_snapshot()

    torch.backends.cudnn.enabled = True

    main()

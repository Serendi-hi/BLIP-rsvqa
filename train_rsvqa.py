import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_vqa import blip_vqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import train_vqa_collate_fn, val_vqa_collate_fn
from data.utils import save_result


def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('train_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,(image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)

        loss = model(image, question, answer, train=True, n=n, weights=weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def val(model, data_loader, device):
    # val
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('val_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))


    for i,(image, question, answer, weights, n) in enumerate(data_loader):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)

        loss = model(image, question, answer, train=True, n=n, weights=weights)

        metric_logger.update(val_loss=loss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Evaluating on test dataset:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []

    if config['inference']=='rank':
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
        answer_candidates.input_ids[:,0] = model.tokenizer.bos_token_id

    for n, (image, question, question_id, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)

        if config['inference']=='generate':
            answers = model(image, question, train=False, inference='generate')

            for answer, ques_id in zip(answers, question_id):
                ques_id = int(ques_id.item())
                result.append({"question_id":ques_id, "answer":answer})

        elif config['inference']=='rank':
            answer_ids = model(image, question, answer_candidates, train=False, inference='rank', k_test=config['k_test'])

            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"question_id":int(ques_id.item()), "answer":answer_list[answer_id]})
    return result

def calculate_acc(vqa_result, test_dataset):
    gt = {}
    for image, question, question_id, answer, type_str in test_dataset:
        gt[question_id] = answer

    n = 0
    n_correct = 0
    for sample in vqa_result:
        n+=1
        index = sample['question_id']
        if sample['answer'].strip() == gt[index]:
                n_correct += 1
    acc = n_correct / n
    print(f"val_acc: {acc}", flush=True)

    return acc


def main(args, config):
    # output_path = os.path.join(args.output_dir, 'lr{}'.format(config['init_lr']))
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_val'],config['batch_size_test']],
                                              num_workers=[4,4,4],is_trains=[True, False, False],
                                              collate_fns=[train_vqa_collate_fn, val_vqa_collate_fn, None])
    #### Model ####
    print("Creating model")
    model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_size'],
                     vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # print(type(config['init_lr']))
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=float(config['init_lr']), weight_decay=config['weight_decay'])

    best_acc = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], float(config['init_lr']), config['min_lr'])

            train_stats = train(model, train_loader, optimizer, epoch, device)
            val_stats = val(model, val_loader, device)
            vqa_result = evaluation(model_without_ddp, test_loader, device, config)
            acc = calculate_acc(vqa_result, test_loader.dataset)
        else:
            break


        if utils.is_main_process():
            log_stats = {'Epoch': epoch,
                         **{f'{k}': v for k, v in train_stats.items()},
                         **{f'{k}': v for k, v in val_stats.items()}
                         }

            with open(os.path.join(args.output_path, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'acc': acc
            }

            if acc >= best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save(save_obj.copy(), os.path.join(args.output_path, 'best_ckpt.pth'))

            print(f"best_acc: {best_acc}", f"best_epoch: {best_epoch}",flush=True)


        # dist.barrier()

    # vqa_result = evaluation(model_without_ddp, test_loader, device, config)
    # calculate_acc(vqa_result, datasets[2])
    # result_file = save_result(vqa_result, args.result_dir, 'vqa_result')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/rsvqa.yaml')
    parser.add_argument('--output_dir', default='output/VQA')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    config_print = config.copy()
    config_print.pop('vqa_root')
    config_print.pop('train_files')
    config_print.pop('ann_root')
    config_print.pop('vit_grad_ckpt')
    config_print.pop('vit_ckpt_layer')
    config_print.pop('min_lr')
    print(config_print)

    args.output_path = os.path.join(args.output_dir, 'lr{}'.format(config['init_lr']))
    args.result_dir = os.path.join(args.output_path, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_path, 'config.yaml'), 'w'))

    main(args, config)
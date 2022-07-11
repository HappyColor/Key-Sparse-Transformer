#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 18:16:12 CST 2021
@author: lab-chen.weidong
"""

import os
from tqdm import tqdm
import torch
import torch.distributed
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import json

import utils
from config import Cfg, create_workshop

class Engine():
    def __init__(self, cfg: Cfg, local_rank: int, world_size: int, free_port: int):
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = world_size
        self.free_port = free_port
        self.workshop = self.cfg.workshop
        self.ckpt_save_path = self.cfg.ckpt_save_path
        self.device = self.cfg.train.device
        self.EPOCH = self.cfg.train.EPOCH
        self.current_epoch = 0
        self.iteration = 0
        self.best_acc = 0
        
        if self.cfg.train.find_init_lr:
            self.cfg.train.lr = 0.000001
            self.cfg.train.step_size = 1
            self.cfg.train.gamma = 1.05
            if self.local_rank == 0:
                self.writer = SummaryWriter(self.cfg.workshop)

        ### prepare model and train tools
        with open('./config/model_config.json', 'r') as f1, open(f'./config/{self.cfg.dataset.database}_feature_config.json', 'r') as f2:
            model_json = json.load(f1)[self.cfg.model.type]
            feas_json = json.load(f2)
            model_json['num_classes'] = feas_json['num_classes']
            self.cfg.train.evaluate = feas_json['evaluate']
            data_json0 = feas_json[self.cfg.dataset.feature[0]]
            data_json1 = feas_json[self.cfg.dataset.feature[1]]
            data_json = {}
            for key in data_json0.keys():
                data_json[key] = [data_json0[key], data_json1[key]]
            data_json['meta_csv_file'] = feas_json['meta_csv_file']
        self.model = utils.model.load_model(self.cfg.model.type, self.device, **model_json)
        self.optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, self.model.parameters()), lr=self.cfg.train.lr, momentum=0.9)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.data_loader_feactory = utils.dataset.DataloaderFactory(self.cfg)
        self.train_dataloader = self.data_loader_feactory.build(state='train', **data_json)
        self.test_dataloader = self.data_loader_feactory.build(state='test', **data_json)
        if self.cfg.train.find_init_lr:
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.train.step_size, gamma=self.cfg.train.gamma)
        else:
            self.scheduler = lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.EPOCH, eta_min=self.cfg.train.lr / 1000)
        
        ### prepare logger
        self.logger_train = utils.logger.create_logger(self.cfg.workshop, name='train') if self.local_rank == 0 else None
        self.logger_test = utils.logger.create_logger(self.cfg.workshop, name='test') if self.local_rank == 0 else None
        if self.logger_train is not None:
            self.logger_train.info(f'workshop: {self.cfg.workshop}')
            self.logger_train.info(f'seed: {self.cfg.train.seed}')
            self.logger_train.info(f'pid: {os.getpid()}')

        ### prepare meters
        self.loss_meter = utils.avgmeter.AverageMeter(device='cpu')
        self.accuracy_meter = utils.avgmeter.AverageMeter(device='cpu')
        self.predict_recoder = utils.recoder.TensorRecorder(device='cpu', dtype=torch.int64)
        self.label_recoder = utils.recoder.TensorRecorder(device='cpu', dtype=torch.int64)
        self.gather_predict = utils.recoder.TensorRecorder(device='cpu', dtype=torch.int64)
        self.gather_label = utils.recoder.TensorRecorder(device='cpu', dtype=torch.int64)
        self.train_acc, self.train_ua, self.train_f1, self.train_loss = [], [], [], []
        self.test_acc, self.test_ua, self.test_f1, self.test_loss = [], [], [], []
        
    def reset_meters(self):
        self.loss_meter.reset()
        self.accuracy_meter.reset()
    
    def reset_recoders(self):
        self.predict_recoder.reset()
        self.label_recoder.reset()
        self.gather_predict.reset()
        self.gather_label.reset()

    def train_epoch(self):
        self.train_dataloader.set_epoch(self.current_epoch)
        if self.local_rank == 0:
            print(f'-------- {self.cfg.workshop} --------')
        discrip_str = f'Epoch-{self.current_epoch}/{self.EPOCH}'
        pbar_train = tqdm(self.train_dataloader, disable=self.local_rank != 0)
        pbar_train.set_description('Train' + discrip_str)

        self.reset_meters()
        self.reset_recoders()

        self.model.train()
        for data in pbar_train:
            self.iteration += 1

            x_a = torch.stack(data[0], dim=0).to(self.device)
            x_t = torch.stack(data[1], dim=0).to(self.device)
            x_a_padding_mask = torch.stack(data[2], dim=0).to(self.device)
            x_t_padding_mask = torch.stack(data[3], dim=0).to(self.device)
            y = torch.tensor(data[4]).to(self.device)

            batch_size = y.shape[0]

            self.optimizer.zero_grad()

            out = self.model(x_a, x_t, x_a_padding_mask, x_t_padding_mask)
            loss = self.loss_func(out, y)

            loss.backward()
            self.optimizer.step()

            y_pred = torch.argmax(out, dim=1).cpu()
            y = y.cpu()

            self.predict_recoder.record(y_pred)
            self.label_recoder.record(y)

            accuracy = utils.toolbox.calculate_accuracy(y_pred, y)
            self.loss_meter.update(loss.item())
            self.accuracy_meter.update(accuracy, batch_size)

            pbar_train_dic = OrderedDict()
            pbar_train_dic['iter'] = self.iteration
            pbar_train_dic['lr'] = self.optimizer.param_groups[0]['lr']
            pbar_train_dic['acc'] = f'{self.accuracy_meter.avg:.5f}'  # acc in local_rank: 0
            pbar_train_dic['loss'] = f'{self.loss_meter.avg:.5f}'    # loss in local_rank: 0
            pbar_train.set_postfix(pbar_train_dic)

            if self.cfg.train.find_init_lr:
                if loss.item() > 20:
                    raise ValueError(f'Loss: {loss.item()} is start to expand, stop the program.')
                if self.local_rank == 0:
                    self.writer.add_scalar('Step Loss', loss.item(), self.iteration)
                    self.writer.add_scalar('Total Loss', self.loss_meter.avg, self.iteration)
                    self.writer.add_scalar('Step LR', self.optimizer.param_groups[0]['lr'], self.iteration)
                self.scheduler.step()

        self.all_recoder_to_file()
        
        if self.local_rank == 0:
            self.gather_world_file()
            epoch_preds = self.gather_predict.data
            epoch_labels = self.gather_label.data
            accuracy, f1, precision, ua, confuse_matrix = utils.toolbox.calculate_score(epoch_preds, epoch_labels)
            self.delete_world_file()

            self.train_acc.append(accuracy)
            self.train_ua.append(ua)
            self.train_f1.append(f1)
            self.train_loss.append(self.loss_meter.avg)

        if self.logger_train is not None:
            self.logger_train.info(
                f'Training epoch: {self.current_epoch}, accuracy: {accuracy:.5f}, precision: {precision:.5f}, recall: {ua:.5f}, F1: {f1:.5f}, loss: {self.loss_meter.avg:.5f}'
            )       
    
    def test(self):
        assert self.test_dataloader is not None, print('Test dataloader is not defined')
        discrip_str = f'Epoch-{self.current_epoch}'
        pbar_test = tqdm(self.test_dataloader, disable=self.local_rank != 0)
        pbar_test.set_description('Test' + discrip_str)

        self.reset_meters()
        self.reset_recoders()

        self.model.eval()
        with torch.no_grad():
            for data in pbar_test:
                x_a = torch.stack(data[0], dim=0).to(self.device)
                x_t = torch.stack(data[1], dim=0).to(self.device)
                x_a_padding_mask = torch.stack(data[2], dim=0).to(self.device)
                x_t_padding_mask = torch.stack(data[3], dim=0).to(self.device)
                y = torch.tensor(data[4]).to(self.device)

                batch_size = y.shape[0]

                out = self.model(x_a, x_t, x_a_padding_mask, x_t_padding_mask)
                loss = self.loss_func(out, y)

                y_pred = torch.argmax(out, dim=1).cpu()
                y = y.cpu()

                self.predict_recoder.record(y_pred)
                self.label_recoder.record(y)

                accuracy = utils.toolbox.calculate_accuracy(y_pred, y)

                self.loss_meter.update(loss.item())
                self.accuracy_meter.update(accuracy, batch_size)

                pbar_test_dic = OrderedDict()
                pbar_test_dic['acc'] = f'{self.accuracy_meter.avg:.5f}'
                pbar_test_dic['loss'] = f'{self.loss_meter.avg:.5f}'
                pbar_test.set_postfix(pbar_test_dic)

            self.all_recoder_to_file()
        
            if self.local_rank == 0:
                self.gather_world_file()
                epoch_preds = self.gather_predict.data
                epoch_labels = self.gather_label.data
                accuracy, f1, precision, ua, confuse_matrix = utils.toolbox.calculate_score(epoch_preds, epoch_labels)
                self.delete_world_file()

                self.test_acc.append(accuracy)
                self.test_ua.append(ua)
                self.test_f1.append(f1)
                self.test_loss.append(self.loss_meter.avg)

                if accuracy > self.best_acc:
                    self.best_acc = accuracy
                    self.model_save(is_best=True)

            if self.logger_test is not None:
                self.logger_test.info(
                    f'Testing epoch: {self.current_epoch}, accuracy: {accuracy:.5f}, precision: {precision:.5f}, recall: {ua:.5f}, F1: {f1:.5f}, loss: {self.loss_meter.avg:.5f}, confuse_matrix: \n{confuse_matrix}'
                )
    
    def all_recoder_to_file(self):
        self.predict_recoder.to_file(f'./temp_files/temp_predict_{self.local_rank}_{self.free_port}.pt')
        self.label_recoder.to_file(f'./temp_files/temp_label_{self.local_rank}_{self.free_port}.pt')
        torch.distributed.barrier()

    def gather_world_file(self):
        for i in range(self.world_size):
            self.gather_predict.record(torch.load(f'./temp_files/temp_predict_{i}_{self.free_port}.pt'))
            self.gather_label.record(torch.load(f'./temp_files/temp_label_{i}_{self.free_port}.pt'))

    def delete_world_file(self):
        for i in range(self.world_size):
            os.remove(f'./temp_files/temp_predict_{i}_{self.free_port}.pt')
            os.remove(f'./temp_files/temp_label_{i}_{self.free_port}.pt')

    def model_save(self, is_best=False):  
        ckpt_save_file = os.path.join(self.ckpt_save_path, 'best.pt') if is_best \
            else os.path.join(self.ckpt_save_path, f'epoch{self.current_epoch}.pt')
        save_dict = {
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
            }
        torch.save(save_dict, ckpt_save_file)

    def run(self):
        if self.cfg.train.find_init_lr:
            while self.current_epoch < self.EPOCH:
                self.train_epoch()
                self.current_epoch += 1
        else:
            while self.current_epoch < self.EPOCH:
                self.train_epoch()
                self.scheduler.step()
                self.test()

                self.current_epoch += 1

                if self.local_rank == 0:
                    plot_data = [self.train_acc, self.train_ua, self.train_f1, self.train_loss, self.test_acc, self.test_ua, self.test_f1, self.test_loss]
                    titles = ['WA-train', 'UA-train', 'F1-train', 'Loss-train', 'WA-test', 'UA-test', 'F1-test', 'Loss-test']
                    utils.write_result.plot_process(plot_data, titles, self.cfg.workshop)

        utils.logger.close_logger(self.logger_train)
        utils.logger.close_logger(self.logger_test)

def main_worker(local_rank, Cfg, world_size, dist_url, free_port):
    utils.environment.set_seed(Cfg.train.seed + local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=dist_url,
        world_size=world_size,
        rank=local_rank,
    )
    
    if Cfg.dataset.database == 'iemocap':
        Cfg.train.strategy = '5cv'
        folds = [1, 2, 3, 4, 5]
    else:
        raise KeyError(f'Unknown database: {Cfg.dataset.database}')

    for f in folds:
        cfg = Cfg.clone()
        cfg.train.current_fold = f
        create_workshop(cfg, local_rank)
        engine = Engine(cfg, local_rank, world_size, free_port)
        engine.run()
        torch.cuda.empty_cache()

    if local_rank == 0:
        criterion = ['accuracy', 'precision', 'recall', 'F1']
        evaluate = cfg.train.evaluate
        outfile = f'result/result_{cfg.model.type}.csv'
        utils.write_result.path_to_csv(os.path.dirname(cfg.workshop), criterion, evaluate, csvfile=outfile)

def main():
    utils.environment.visible_gpus(Cfg.train.device_id)
    utils.environment.set_seed(Cfg.train.seed)

    free_port = utils.distributed.find_free_port()
    dist_url = f'tcp://127.0.0.1:{free_port}'   
    world_size = torch.cuda.device_count()    # num_gpus
    print(f'world_size={world_size} Using dist_url={dist_url}')

    mp.spawn(fn=main_worker, args=(Cfg, world_size, dist_url, free_port), nprocs=world_size)

if __name__=='__main__':
    main()
    

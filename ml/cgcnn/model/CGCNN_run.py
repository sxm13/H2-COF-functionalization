from model.CGCNN_model import CrystalGraphConvNet,Normalizer
from model.CGCNN_data import collate_pool,get_train_val_test_loader,CIFData
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import shutil
import numpy as np
import csv

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))
def mse(prediction, target):
    return torch.mean(torch.square(target - prediction))

class FineTune(object):
    def __init__(self, root_dir, log_dir,log_every_n_steps=50, eval_every_n_epochs = 1, epoch = 500,
                  opti ="SGD", lr = 0.001, momentum = 0.9, weight_decay = 1e-6, dataset= "COF_H2.csv", 
                  n_out = 3, batch_size = 32,n_conv=3, random_seed = 1029, pin_memory=False):
        self.n_conv = n_conv
        self.root_dir = root_dir
        self.n_out = n_out
        self.data = dataset
        self.eval_every_n_epochs = eval_every_n_epochs
        self.log_every_n_steps = log_every_n_steps
        self.epochs = epoch
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.opti = opti
        self.lr = lr
        self.writer = SummaryWriter(log_dir=log_dir)
        self.criterion = nn.MSELoss()
        self.root_dir = root_dir
        self.dataset = CIFData(root_dir = self.root_dir, data_file = self.data, len = self.n_out)
        self.random_seed = random_seed
        collate_fn = collate_pool
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.device = self._get_device()
        self.model_checkpoints_folder = log_dir + "checkpoints/"
        
        self.train_loader, self.valid_loader, self.test_loader = get_train_val_test_loader(
            dataset = self.dataset,
            random_seed = self.random_seed,
            collate_fn = collate_fn,
            pin_memory = self.pin_memory,
            batch_size = self.batch_size, 
        )
        sample_data_list = [self.dataset[i] for i in range(len(self.dataset))]
        _, sample_target, _ = collate_pool(sample_data_list)
        self.normalizer = Normalizer(sample_target)

    def _get_device(self):
        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.set_device(0)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def train(self):
        structures, _, _ = self.dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,n_conv = self.n_conv,n_out=self.n_out)

        if self.device == 'cuda':
            torch.cuda.set_device(0)
            model.to(self.device)
            print("Use cuda for torch")
        else:
            print("Only use cpu for torch")
        layer_list = []
        for name, param in model.named_parameters():
            if 'fc_out' in name:
                print(name, 'new layer')
                layer_list.append(name)
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))
        if self.opti == 'SGD':
            optimizer = optim.SGD(
                [{'params': base_params, 'lr': self.lr}, {'params': params}],
                 self.lr, momentum=self.momentum, 
                weight_decay=self.weight_decay
            )
        elif self.opti == 'Adam':
            lr_multiplier = 0.2
            optimizer = optim.Adam(
                [{'params': base_params, 'lr': self.lr*lr_multiplier}, {'params': params}],
                self.lr, weight_decay=self.weight_decay
            )
        else:
            raise NameError('Only SGD or Adam is allowed as optimizer')        
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_mae = np.inf
        best_valid_roc_auc = 0
        for epoch_counter in range(self.epochs):
            for bn, (input, target, _) in enumerate(self.train_loader):
                if self.device == 'cuda':
                    input_var = (Variable(input[0].to(self.device, non_blocking=True)),
                                Variable(input[1].to(self.device, non_blocking=True)),
                                input[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3])
                target_normed = self.normalizer.norm(target)
                if self.device == 'cuda':
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)
                output = model(*input_var)
                loss = self.criterion(output, target_var)
                if bn % self.log_every_n_steps == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    print('Epoch: %d, Batch: %d, Loss:'%(epoch_counter+1, bn), loss.item())
                    print('Epoch: %d' % (epoch_counter + 1))
                    print('train:', loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1
            if epoch_counter % self.eval_every_n_epochs == 0:
                valid_loss, valid_mae = self._validate(model, self.valid_loader, epoch_counter)
                if valid_mae < best_valid_mae:
                    best_valid_mae = valid_mae
                    torch.save(model.state_dict(), os.path.join(self.model_checkpoints_folder, 'model.pth'))
                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
        self.model = model
    
    def _validate(self, model, valid_loader, n_epoch):
        losses = AverageMeter()
        mae_errors = AverageMeter()
        with torch.no_grad():
            model.eval()
            for bn, (input, target, _) in enumerate(valid_loader):
                if self.device == 'cuda':
                    input_var = (Variable(input[0].to(self.device, non_blocking=True)),
                                Variable(input[1].to(self.device, non_blocking=True)),
                                input[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                Variable(input[1]),
                                input[2],
                                input[3])
                target_normed = self.normalizer.norm(target)
                if self.device == 'cuda':
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)
                output = model(*input_var)
                loss = self.criterion(output, target_var)
                mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
            print('Epoch [{0}] Validate: [{1}/{2}],''Loss {loss.val:.4f} ({loss.avg:.4f}),''MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                n_epoch+1, bn+1, len(self.valid_loader), loss=losses, mae_errors=mae_errors))
        model.train()
        print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return losses.avg, mae_errors.avg

    def test(self):

        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        print(model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print("Loaded trained model with success.")
        losses = AverageMeter()
        mae_errors = AverageMeter()
        test_targets = []
        test_preds = []
        test_cif_ids = []
        with torch.no_grad():
            self.model.eval()
            for bn, (input, target, batch_cif_ids) in enumerate(self.test_loader):
                if self.device == 'cuda':
                    input_var = (Variable(input[0].to(self.device, non_blocking=True)),
                                Variable(input[1].to(self.device, non_blocking=True)),
                                input[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                            Variable(input[1]),
                            input[2],
                            input[3])
                target_normed = self.normalizer.norm(target)
                if self.device == 'cuda':
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)
                output = self.model(*input_var)
                loss = self.criterion(output, target_var)
                mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
                test_pred = self.normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
            print('Test: [{0}/{1}], '
                    'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                    'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                bn, len(self.valid_loader), loss=losses,
                mae_errors=mae_errors))

        with open(os.path.join(self.writer.log_dir, 'test_names.csv'), 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id))
        self.model.train()

        print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return losses.avg, mae_errors.avg

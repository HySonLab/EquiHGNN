import os
import time
import argparse
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from models import *
from data import OneTarget
from utils import create_model, create_data

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, CometLogger
from torchmetrics.wrappers import BootStrapper
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics import MetricCollection

class LitModel(pl.LightningModule):
    def __init__(self, hparams, std=None):
        super(LitModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.std = std

        # Initialize the model based on the method
        model_cls = create_model(model_name=self.hparams.method)
        if model_cls.__name__ == "GNN_2D":
            self.model = model_cls(1, gnn_type=self.hparams.method, drop_ratio=self.hparams.dropout)
        else:
            self.model = model_cls(1, self.hparams)
        
        self.mse_loss_fn = nn.MSELoss()
        self.eval_metrics = MetricCollection(
            {
            "mae": BootStrapper(
                base_metric=MeanAbsoluteError(),
                num_bootstraps=50
                ),
            "mse": BootStrapper(
                base_metric=MeanSquaredError(),
                num_bootstraps=50
                ),
            }
        )

    def forward(self, data):
        return self.model(data)

    def training_step(self, data, batch_idx):
        out = self(data)
        mse_loss = self.mse_loss_fn(out, data.y)

        self.log('train_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        
        return mse_loss

    def validation_step(self, data, batch_idx):
        out = self(data)
        if self.std:
            self.eval_metrics.update(out * self.std, data.y * self.std)
        else:
            self.eval_metrics.update(out, data.y)

    def on_validation_epoch_end(self):
        mae_loss = self.eval_metrics.compute()
        mae_loss = {f"val_{k}": v for k, v in mae_loss.items()}
        self.log_dict(mae_loss, prog_bar=True)

        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', float(f"{lr:.5e}"), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)

        self.eval_metrics.reset()


    def test_step(self, data, batch_idx):
        out = self(data)
        if self.std:
            self.eval_metrics.update(out * self.std, data.y * self.std)
        else:
            self.eval_metrics.update(out, data.y)

    def on_test_epoch_end(self):
        mae_loss = self.eval_metrics.compute()
        mae_loss = {f"test_{k}": v for k, v in mae_loss.items()}
        self.log_dict(mae_loss)

        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', float(f"{lr:.5e}"), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)

        self.eval_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=self.hparams.min_lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_mae_mean'
            }
        }

if __name__ == '__main__':
    print('Task start time:')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_time = time.time()

    parser = argparse.ArgumentParser(description='OPV Training with MHNN')

    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default="datasets/opv3d")
    parser.add_argument('--target', type=int, default=0, help='target of dataset')
    parser.add_argument('--data', default='opv_hg', help='data type')
    parser.add_argument('--use_ring', action="store_true", help='using rings with conjugated bonds')

    # Training hyperparameters
    parser.add_argument('--runs', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--min_lr', default=0.000001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--clip_gnorm', default=None, type=float)

    # Model hyperparameters
    parser.add_argument('--method', default='mhnns', help='model type')
    parser.add_argument('--All_num_layers', default=3, type=int, help='number of basic blocks')
    parser.add_argument('--MLP1_num_layers', default=2, type=int, help='layer number of mlps')
    parser.add_argument('--MLP2_num_layers', default=2, type=int, help='layer number of mlp2')
    parser.add_argument('--MLP3_num_layers', default=2, type=int, help='layer number of mlp3')
    parser.add_argument('--MLP4_num_layers', default=2, type=int, help='layer number of mlp4')
    parser.add_argument('--MLP_hidden', default=64, type=int, help='hidden dimension of mlps')
    parser.add_argument('--output_num_layers', default=2, type=int)
    parser.add_argument('--output_hidden', default=64, type=int)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    parser.add_argument('--normalization', default='ln', choices=['bn', 'ln', 'None'])
    parser.add_argument('--activation', default='relu', choices=['Id', 'relu', 'prelu'])
    parser.add_argument('--dropout', default=0.0, type=float)

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load dataset and normalize targets to mean = 0 and std = 1
    if args.target in [0, 1, 2, 3]:
        args.polymer = False
    elif args.target in [4, 5, 6, 7]:
        args.polymer = True
    else:
        raise Exception('Invalid target value!')

    transform = T.Compose([OneTarget(target=args.target)])

    data_cls = create_data(data_name=args.data)
    
    train_dataset = data_cls(root=args.data_dir, polymer=args.polymer, partition='train', transform=transform, use_ring=args.use_ring)
    valid_dataset = data_cls(root=args.data_dir, polymer=args.polymer, partition='valid', transform=transform, use_ring=args.use_ring)
    test_dataset = data_cls(root=args.data_dir, polymer=args.polymer, partition='test', transform=transform, use_ring=args.use_ring)

    # Normalize targets to mean = 0 and std = 1.
    mean = train_dataset._data.y.mean(dim=0, keepdim=True)
    std = train_dataset._data.y.std(dim=0, keepdim=True)
    train_dataset._data.y = (train_dataset._data.y - mean) / std
    valid_dataset._data.y = (valid_dataset._data.y - mean) / std
    test_dataset._data.y = (test_dataset._data.y - mean) / std
    mean, std = mean[:, args.target].item(), std[:, args.target].item()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Set up loggers
    csv_logger = CSVLogger(save_dir='logs/', name="opv_" + str(args.target) + "_" + args.method + "_" + args.data)
    commet_logger = CometLogger(
        api_key=os.environ["COMET_API_KEY"] if "COMET_API_KEY" in os.environ else None,
        project_name="Geometric Molecular Hypergraph",
        experiment_name=f"opv_{args.target}_{args.method}_{args.data}",
        save_dir="logs/"
    )
    loggers = [
        commet_logger, 
        csv_logger
        ]
    
    # torch.set_float32_matmul_precision("medium")

    summary_callback = pl.callbacks.ModelSummary(max_depth=8)
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_mae_mean',
        patience=5,
        verbose=True,
        mode='min',  
        )
    
    callbacks = [summary_callback, early_stop_callback]

    for run in range(args.runs):
        # Set global seed for this run
        seed = args.seed + run
        pl.seed_everything(seed=seed, workers=True)
        print(f'\nRun No. {run+1}:')
        print(f'Seed: {seed}\n')

        # Initialize Lightning model
        model = LitModel(args, std=std)

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            logger=loggers,
            callbacks=callbacks,
            devices=1 if torch.cuda.is_available() else 0,
        )

        trainer.fit(model, train_loader, valid_loader)
        trainer.test(model, test_loader, ckpt_path="best")

    print('Task end time:')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    end_time = time.time()
    print('Total time taken: {} s.'.format(int(end_time - start_time)))

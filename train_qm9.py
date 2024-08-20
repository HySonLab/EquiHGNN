import time
import argparse
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from models import MHNN, GNN_2D, MHNNS
from data import QM9HGraph, QM9HGraph3D, OneTarget

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger

class LitModel(pl.LightningModule):
    def __init__(self, hparams, std=None):
        super(LitModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.std = std
        # Initialize the model based on the method
        if self.hparams.method == 'mhnn':
            self.model = MHNN(1, self.hparams)
        elif self.hparams.method == 'mhnns':
            self.model = MHNNS(1, self.hparams)
        elif self.hparams.method in ['gin', 'gcn', 'gat', 'gatv2']:
            self.model = GNN_2D(1, gnn_type=self.hparams.method, drop_ratio=self.hparams.dropout)
        else:
            raise ValueError(f'Undefined model name: {self.hparams.method}')
        
        self.mse_loss_fn = nn.MSELoss()
        self.mae_loss_fn = nn.L1Loss()

    def forward(self, data):
        return self.model(data)

    def training_step(self, data, batch_idx):
        out = self(data)
        mse_loss = self.mse_loss_fn(out, data.y)
        # mae_loss = self.mae_loss_fn(out, data.y)

        self.log('train_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        # self.log('train_mae', mae_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        
        return mse_loss

    def validation_step(self, data, batch_idx):
        out = self(data)
        mae_loss = self.mae_loss_fn(out, data.y)
        if self.std:
            mae_loss = mae_loss * self.std
        self.log('val_mae', mae_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)

        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', float(f"{lr:.5e}"), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        
        return mae_loss

    def test_step(self, data, batch_idx):
        out = self(data)
        mae_loss = self.mae_loss_fn(out, data.y)
        if self.std:
            mae_loss = mae_loss * self.std
        self.log('test_mae', mae_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        
        return mae_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=self.hparams.min_lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

if __name__ == '__main__':
    print('Task start time:')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_time = time.time()

    parser = argparse.ArgumentParser(description='OPV Training with MHNN')

    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default="datasets/qm9")
    parser.add_argument('--target', type=int, default=0, help='target of dataset')

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
    parser.add_argument('--method', default='mhnn', help='model type')
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

    transform = T.Compose([OneTarget(target=args.target)])

    dataset = QM9HGraph(root=args.data_dir, transform=transform)

    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    # Calculate the number of samples in each set
    train_size = int(train_ratio * len(dataset))
    valid_size = int(valid_ratio * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    # Split the dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # Normalize targets to mean = 0 and std = 1.
    mean = train_dataset.dataset.data.y[train_dataset.indices].mean(dim=0, keepdim=True)
    std = train_dataset.dataset.data.y[train_dataset.indices].std(dim=0, keepdim=True)

    train_dataset.dataset.data.y[train_dataset.indices] = (train_dataset.dataset.data.y[train_dataset.indices] - mean) / std
    valid_dataset.dataset.data.y[valid_dataset.indices] = (valid_dataset.dataset.data.y[valid_dataset.indices] - mean) / std
    test_dataset.dataset.data.y[test_dataset.indices] = (test_dataset.dataset.data.y[test_dataset.indices] - mean) / std

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Set up CSV logger
    csv_logger = CSVLogger(save_dir='logs/', name=args.data_dir + "_" + str(args.target))
    wandb_logger = WandbLogger(
    project="Geometric Molecular Hypergraph",
    name=f"opv_{args.target}_{args.method}",
    save_dir="logs/"
    )
    loggers = [
        wandb_logger, 
        csv_logger
        ]
    torch.set_float32_matmul_precision("medium")

    summary_callback = pl.callbacks.ModelSummary(max_depth=8)
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
        )
    callbacks = [summary_callback]

    for run in range(args.runs):
        # Set global seed for this run
        seed = args.seed + run
        pl.seed_everything(seed=seed, workers=True)
        print(f'\nRun No. {run+1}:')
        print(f'Seed: {seed}\n')

        # Initialize Lightning model
        model = LitModel(args)

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            logger=csv_logger,
            callbacks=callbacks,
            devices=1 if torch.cuda.is_available() else 0,
        )

        trainer.fit(model, train_loader, valid_loader)
        trainer.test(model, test_loader)

    print('Task end time:')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    end_time = time.time()
    print('Total time taken: {} s.'.format(int(end_time - start_time)))

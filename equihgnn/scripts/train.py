import argparse
import os
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn

from dotenv import load_dotenv

from pytorch_lightning.loggers import CometLogger, WandbLogger, MLFlowLogger, CSVLogger

from torch_geometric.loader import DataLoader
from torchmetrics.regression import MeanAbsoluteError

from equihgnn.utils import create_model, create_train_val_test_set_and_normalize, Config

torch.set_float32_matmul_precision("medium")

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
comet_api_key = os.getenv("COMET_API_KEY")
wandb_api_key = os.getenv("WANDB_API_KEY")
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

class PL_Model(pl.LightningModule):
    def __init__(self, hparams, std=None):
        super(PL_Model, self).__init__()
        self.save_hyperparameters(hparams)
        self.std = std

        # Initialize the model based on the method
        model_cls = create_model(model_name=self.hparams.method)
        if model_cls.__name__ == "GNN_2D":
            self.model = model_cls(
                1, gnn_type=self.hparams.method, drop_ratio=self.hparams.dropout
            )
        else:
            self.model = model_cls(1, self.hparams)

        self.mse_loss_fn = nn.MSELoss()
        self.mae_metric = MeanAbsoluteError()

    def forward(self, data):
        return self.model(data)

    def training_step(self, data, batch_idx):
        out = self(data)
        mse_loss = self.mse_loss_fn(out, data.y)

        self.log(
            "train_loss",
            mse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
        )

        return mse_loss

    def validation_step(self, data, batch_idx):
        out = self(data)
        if self.std:
            self.mae_metric.update(out * self.std, data.y * self.std)
        else:
            self.mae_metric.update(out, data.y)

    def on_validation_epoch_end(self):
        mae_loss = self.mae_metric.compute()
        self.log("val_mae", mae_loss, prog_bar=True, sync_dist=True)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log(
            "lr",
            float(f"{lr:.5e}"),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
        )

        self.mae_metric.reset()

    def test_step(self, data, batch_idx):
        out = self(data)
        if self.std:
            self.mae_metric.update(out * self.std, data.y * self.std)
        else:
            self.mae_metric.update(out, data.y)

    def on_test_epoch_end(self):
        mae_loss = self.mae_metric.compute()
        self.log("test_mae", mae_loss, sync_dist=True)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log(
            "lr",
            float(f"{lr:.5e}"),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
        )

        self.mae_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, 
            min_lr=self.hparams.min_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_mae"},
        }


def main():
    print("Task start time:")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Training with Configurable YAML")
    parser.add_argument("configs", type=str, help="Path to config file")  # Positional argument for yaml config
    parser.add_argument("--target", type=int, help="Override target value in config")
    parser.add_argument("--exp-suffix", type=str, default="", help="Experiment Suffix")

    args = parser.parse_args()

    # Load configuration from YAML file
    configs = Config.from_yaml(args.configs)
    if args.target is not None:
        configs.target = args.target

    device = f"cuda:{configs.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset and normalize targets to mean = 0 and std = 1
    (
        train_dataset,
        valid_dataset,
        test_dataset,
        std,
    ) = create_train_val_test_set_and_normalize(
        target=configs.target,
        data_name=configs.data,
        data_dir=configs.data_dir,
        use_ring=configs.use_ring,
    )

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=configs.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=configs.num_workers)

    for run in range(configs.runs):
        # Set global seed for this run
        seed = configs.seed + run
        pl.seed_everything(seed=seed, workers=True)
        print(f"\nRun No. {run+1}:")
        print(f"Seed: {seed}\n")
        
        # Set up loggers
        suffix = "_use_ring" if configs.use_ring else "_no_ring"
        experiment_name = f"{configs.data}_{configs.target}_{configs.method}_{suffix}_{args.exp_suffix}"
        csv_logger = CSVLogger(
            save_dir="logs/",
            name=experiment_name
        )
        experiment_save_dir = os.path.join("logs", experiment_name, f"version_{csv_logger.version}")
        os.makedirs(experiment_save_dir, exist_ok=True)

        loggers = [csv_logger]

        # Add CometLogger if API key is available
        if comet_api_key:
            comet_logger = CometLogger(
                api_key=comet_api_key,
                project_name="equivariant-hypergraph-neural-network",
                experiment_name=experiment_name,
                save_dir=experiment_save_dir,
            )
            loggers.append(comet_logger)

        # Add WandBLogger if API key is available
        if wandb_api_key:
            wandb_logger = WandbLogger(
                project="equivariant-hypergraph-neural-network",
                name=experiment_name,
                save_dir=experiment_save_dir,
            )
            loggers.append(wandb_logger)

        # Add MLFlowLogger if tracking URI is available
        if mlflow_tracking_uri:
            mlflow_logger = MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=mlflow_tracking_uri,
            )
            loggers.append(mlflow_logger)

        # Initialize Lightning model
        model = PL_Model(configs, std=std)

        summary_callback = pl.callbacks.ModelSummary(max_depth=8)
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=20,
            # verbose=True, 
            mode='min'
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=experiment_save_dir,
            filename="{epoch}-{val_mae_mean}",
            save_top_k=1,
            monitor='val_mae',
            mode='min'
        )

        callbacks = [summary_callback, early_stopping_callback, checkpoint_callback ]

        trainer_args = {
            "max_epochs": configs.epochs,
            "logger": loggers,
            "devices": "auto",
            "callbacks": callbacks,
        }
        if configs.debug:
            trainer_args["fast_dev_run"] = True

        trainer = pl.Trainer(**trainer_args
                            #  , strategy="ddp_find_unused_parameters_true"
                             )
        trainer.fit(model, train_loader, valid_loader)

        test_args = {
            "model": model,
            "dataloaders": test_loader,
        }
        if not configs.debug:
            test_args["ckpt_path"] = "best"

        trainer.test(**test_args)

    print("Task end time:")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    end_time = time.time()
    print("Total time taken: {} s.".format(int(end_time - start_time)))

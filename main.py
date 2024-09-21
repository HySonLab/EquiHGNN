import argparse
import os
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, CSVLogger
from torch_geometric.loader import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.wrappers import BootStrapper

from equihgnn.utils import create_model, create_train_val_test_set_and_normalize

torch.set_float32_matmul_precision("medium")


class LitModel(pl.LightningModule):
    def __init__(self, hparams, std=None):
        super(LitModel, self).__init__()
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
        self.eval_metrics = MetricCollection(
            {
                "mae": BootStrapper(base_metric=MeanAbsoluteError(), num_bootstraps=50),
                "mse": BootStrapper(base_metric=MeanSquaredError(), num_bootstraps=50),
            }
        )

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
            self.eval_metrics.update(out * self.std, data.y * self.std)
        else:
            self.eval_metrics.update(out, data.y)

    def on_validation_epoch_end(self):
        mae_loss = self.eval_metrics.compute()
        mae_loss = {f"val_{k}": v for k, v in mae_loss.items()}
        self.log_dict(mae_loss, prog_bar=True, sync_dist=True)

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
        self.log_dict(mae_loss, sync_dist=True)

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

        self.eval_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.25, patience=5, 
            # min_lr=self.hparams.min_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_mae_mean"},
        }


if __name__ == "__main__":
    print("Task start time:")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Training with MHNN")

    # Dataset arguments
    parser.add_argument("--data_dir", type=str, default="datasets/opv3d")
    parser.add_argument("--target", type=int, help="target of dataset", required=True)
    parser.add_argument("--data", default="opv_hg", help="data type")
    parser.add_argument(
        "--use_ring", action="store_true", help="using rings with conjugated bonds"
    )

    # Training hyperparameters
    parser.add_argument("--runs", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--min_lr", default=0.000001, type=float)
    parser.add_argument("--wd", default=0.0, type=float)
    parser.add_argument("--clip_gnorm", default=None, type=float)

    # Model hyperparameters
    parser.add_argument("--method", default="mhnns", help="model type")
    parser.add_argument(
        "--All_num_layers", default=3, type=int, help="number of basic blocks"
    )
    parser.add_argument(
        "--MLP1_num_layers", default=2, type=int, help="layer number of mlps"
    )
    parser.add_argument(
        "--MLP2_num_layers", default=2, type=int, help="layer number of mlp2"
    )
    parser.add_argument(
        "--MLP3_num_layers", default=2, type=int, help="layer number of mlp3"
    )
    parser.add_argument(
        "--MLP4_num_layers", default=2, type=int, help="layer number of mlp4"
    )
    parser.add_argument(
        "--MLP_hidden", default=64, type=int, help="hidden dimension of mlps"
    )
    parser.add_argument("--output_num_layers", default=2, type=int)
    parser.add_argument("--output_hidden", default=64, type=int)
    parser.add_argument("--aggregate", default="mean", choices=["sum", "mean"])
    parser.add_argument("--normalization", default="ln", choices=["bn", "ln", "None"])
    parser.add_argument("--activation", default="relu", choices=["Id", "relu", "prelu"])
    parser.add_argument("--dropout", default=0.0, type=float)

    # Debugging
    parser.add_argument(
        "--debug", action="store_true", help="Debug by forwarding one step only"
    )

    args = parser.parse_args()
    print(args)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset and normalize targets to mean = 0 and std = 1
    (
        train_dataset,
        valid_dataset,
        test_dataset,
        std,
    ) = create_train_val_test_set_and_normalize(
        target=args.target,
        data_name=args.data,
        data_dir=args.data_dir,
        use_ring=args.use_ring,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    for run in range(args.runs):
        # Set global seed for this run
        seed = args.seed + run
        pl.seed_everything(seed=seed, workers=True)
        print(f"\nRun No. {run+1}:")
        print(f"Seed: {seed}\n")
        
        # Set up loggers
        suffix = "_use_ring" if args.use_ring else "_no_ring"
        experiment_name = f"{args.data}_{args.target}_{args.method}_{suffix}"
        csv_logger = CSVLogger(
            save_dir="logs/",
            name=experiment_name
        )
        experiment_save_dir = os.path.join("logs", experiment_name, f"version_{csv_logger.version}")

        commet_logger = CometLogger(
            api_key=os.environ["COMET_API_KEY"] if "COMET_API_KEY" in os.environ else None,
            project_name="Geometric Molecular Hypergrartyrtph",
            experiment_name=experiment_name,
            save_dir=experiment_save_dir,
        )
        loggers = [commet_logger, csv_logger]

        # Initialize Lightning model
        model = LitModel(args, std=std)

        summary_callback = pl.callbacks.ModelSummary(max_depth=8)
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor='val_mae_mean',
            patience=20,
            # verbose=True, 
            mode='min'
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=experiment_save_dir,
            filename="{epoch}-{val_mae_mean}",
            save_top_k=1,
            monitor='val_mae_mean',
            mode='min'
        )

        callbacks = [summary_callback, early_stopping_callback, checkpoint_callback ]

        trainer_args = {
            "max_epochs": args.epochs,
            "logger": loggers,
            "devices": "auto",
            "callbacks": callbacks,
        }
        
        if args.debug:
            trainer_args["fast_dev_run"] = True

        trainer = pl.Trainer(**trainer_args, strategy="ddp_find_unused_parameters_true")

        trainer.fit(model, train_loader, valid_loader)

        test_args = {
            "model": model,
            "dataloaders": test_loader,
        }

        if not args.debug:
            test_args["ckpt_path"] = "best"

        trainer.test(**test_args)

    print("Task end time:")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    end_time = time.time()
    print("Total time taken: {} s.".format(int(end_time - start_time)))

import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data import GTSRBData
from module import GTSRBModule


def main(args):

    if bool(args.download_weights):
        GTSRBData.download_weights()
    else:
        seed_everything(0)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        logger = False
        if args.logger == "wandb":
            logger = WandbLogger(name=args.classifier, project="gtsrb")
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger("gtsrb", name=args.classifier)

        checkpoint = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_last=False,
            filename="{epoch:02d}-{val_acc:.2f}",
            auto_insert_metric_name=False,
        )

        trainer = Trainer(
            fast_dev_run=bool(args.dev),
            logger=logger if not bool(args.dev + args.test_phase) else False,
            accelerator="gpu",
            devices=-1,
            deterministic=True,
            enable_model_summary=False,
            log_every_n_steps=1,
            max_epochs=args.max_epochs,
            callbacks=[checkpoint],
            precision=args.precision,
        )

        model = GTSRBModule(args)
        data = GTSRBData(args)

        if bool(args.pretrained):
            state_dict = os.path.join(
                "cifar10_models", "state_dicts", args.classifier + ".pt"
            )
            model.model.load_state_dict(torch.load(state_dict))

        if bool(args.test_phase):
            trainer.test(model, datamodule=data)
        else:
            trainer.fit(model, datamodule=data)
            if not bool(args.dev):
                trainer.test(model, datamodule=data, ckpt_path="best")


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/data/huy/cifar10")
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "gtsrb"],
        help="Dataset used for validation/test and best-checkpoint selection.",
    )
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default="../gtsrb_synthetic_dataset/data/synthetic_gtsrb",
        help="Synthetic dataset root. Can point to the root containing manifest.csv or directly to images/.",
    )
    parser.add_argument(
        "--use_synthetic_train",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to concatenate synthetic training samples with the real GTSRB training split.",
    )
    parser.add_argument(
        "--synthetic_train_data_dir",
        type=str,
        default="../gtsrb_synthetic_dataset/data/synthetic_gtsrb_train",
        help="Synthetic training dataset root. Can point to the root containing manifest.csv or directly to images/.",
    )
    parser.add_argument(
        "--eval_drop_last",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether validation/test dataloaders drop the last incomplete batch.",
    )
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="none", choices=["none", "tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0,1")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    args = parser.parse_args()
    main(args)

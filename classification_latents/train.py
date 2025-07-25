import json
import os
import warnings
import wandb
import torch

from absl import app, flags
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold  # Importer KFold pour la validation croisée

from classification_latents.classification_dataset import ClassificationDataset
from classification_latents.parse_parameters import get_num_workers
from classification_latents.lightning_module import LightningModuleClassification
from classification_latents.utils.progressbar import ColabProgressBar

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

torch.set_float32_matmul_precision("high")

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "my_project", "Name of the WandB project")
flags.DEFINE_string("project_folder", "pn_features", "Name of the WandB project")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
flags.DEFINE_float("init_lr", 0.001, "Initial learning rate")
flags.DEFINE_integer("batch_size", 2, "Batch size")
flags.DEFINE_integer("warmup", 1000, "Learning rate warmup steps")
flags.DEFINE_integer("num_workers", 2, "Number of DataLoader workers")
flags.DEFINE_integer("prefetch_factor", 2, "Number of batches to prefetch")
flags.DEFINE_string("model_save_path", None, "Path to the checkpoint (.ckpt) file")
flags.DEFINE_bool("no_edge_feature", False, "Whether to use edge features")
flags.DEFINE_string(
    "training_parameters_path", None, "Path to the training parameters JSON file"
)


def main(argv):
    del argv

    # Check that the training parameters path is provided
    if not FLAGS.training_parameters_path:
        logger.error("The 'training_parameters_path' flag must be provided.")
        return

    # Load training parameters from JSON file
    training_parameters_path = FLAGS.training_parameters_path
    logger.info(f"Opening training parameters from {training_parameters_path}")
    try:
        with open(training_parameters_path, "r") as fp:
            parameters = json.load(fp)
    except Exception as e:
        logger.error(f"Error reading training parameters: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_project_name = FLAGS.project_name
    wandb_project_folder = FLAGS.project_folder
    num_epochs = FLAGS.num_epochs
    initial_lr = FLAGS.init_lr
    batch_size = FLAGS.batch_size
    warmup = FLAGS.warmup
    num_workers = FLAGS.num_workers
    prefetch_factor = FLAGS.prefetch_factor
    model_save_path = FLAGS.model_save_path
    use_edge_feature = not FLAGS.no_edge_feature


    dataset = ClassificationDataset(
        root_folder=parameters["dataset"]["obj_folder"],
        meta_path=parameters["dataset"]["meta_path"],
        processing=parameters["dataset"]["processing"],
        number_of_samples=parameters["dataset"]["number_of_samples"],
        number_of_connections=parameters["dataset"]["number_of_connections"],
        model_type=parameters["model"]["type"],
    )

    num_workers = get_num_workers(param=parameters, default_num_workers=num_workers)

    kf = KFold(n_splits=5, shuffle=True)

    fold_idx = 0

    for train_idx, val_idx in kf.split(dataset):
        logger.info(f"Training fold {fold_idx + 1}/5 for {wandb_project_name}")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_dataloader_kwargs = {
            "dataset": train_subset,
            "shuffle": True,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "drop_last": True,
        }

        valid_dataloader_kwargs = {
            "dataset": val_subset,
            "shuffle": False,
            "batch_size": 1,
            "num_workers": num_workers,
        }

        # Update arguments if num_workers > 0
        if num_workers > 0:
            train_dataloader_kwargs.update(
                {
                    "prefetch_factor": prefetch_factor,
                    "persistent_workers": True,
                }
            )
            valid_dataloader_kwargs.update(
                {
                    "prefetch_factor": prefetch_factor,
                    "persistent_workers": True,
                }
            )
    

        # Create DataLoaders
        train_dataloader = DataLoader(**train_dataloader_kwargs)
        valid_dataloader = DataLoader(**valid_dataloader_kwargs)

        # Define or resume model
        num_steps = num_epochs * len(train_dataloader)

        if model_save_path and os.path.isfile(model_save_path):
            logger.info(f"Loading model from checkpoint: {model_save_path}")
            lightning_module = LightningModuleClassification.load_from_checkpoint(
                checkpoint_path=model_save_path,
                parameters=parameters,
                warmup=warmup,
                learning_rate=initial_lr,
                num_steps=num_steps,
            )
        else:
            logger.info("Initializing new model")
            lightning_module = LightningModuleClassification(
                parameters=parameters,
                learning_rate=initial_lr,
                num_steps=num_steps,
                warmup=warmup,
            )

        # Initialize WandbLogger
        wandb_run = wandb.init(
            project=wandb_project_folder,
            name=f"{wandb_project_name}_fold_{fold_idx + 1}",
            tags=[f"fold_{fold_idx + 1}", f"{wandb_project_name}"],
            group=f"{wandb_project_name}_5fold",
        )
        wandb_logger = WandbLogger(experiment=wandb_run)
        checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/")
        lr_monitor = LearningRateMonitor(logging_interval="step")

        wandb_logger.experiment.config.update(
            {
                "architecture": parameters["model"]["type"],
                "#_layers": parameters["model"]["hidden_layers"],
                "#_neurons": parameters["model"]["hidden_size"],
                "#_hops": parameters["dataset"]["khop"],
                "max_lr": initial_lr,
                "batch_size": batch_size,
                "dim_model": parameters["model"]["dim_model"],
            }
        )

        # Configure Trainer
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=num_epochs,
            logger=wandb_logger,
            callbacks=[
                ColabProgressBar(),
                checkpoint_callback,
                lr_monitor,
                # early_stopping_callback,
            ],
            log_every_n_steps=10,
        )

        # Start training
        logger.success(f"Starting training for fold {fold_idx + 1}")
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )

        fold_idx += 1
        wandb_run.finish()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)

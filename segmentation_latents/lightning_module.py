import lightning as L
import torch
from segmentation_latents.model import SegmentationPointNetP2, PointNetSegmenter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import wandb
from torch_geometric.data import Batch
from segmentation_latents.utils.scheduler import CosineWarmupScheduler
from sklearn.model_selection import KFold

class LightningModuleClassification(L.LightningModule):
    def __init__(
        self,
        parameters: dict,
        learning_rate: float,
        num_steps: int,
        warmup: int,
        k_folds: int = 5,  # Paramètre pour la validation croisée
    ):
        super().__init__()
        self.save_hyperparameters()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.param = parameters
        self.k_folds = k_folds  # Utilisation du nombre de plis pour la validation croisée

        model_type = parameters["model"]["type"]

        model_kwargs ={
            "node_input_size": parameters["model"]["node_input_size"],
            "dim_model": parameters["model"]["dim_model"],
            "output_size": parameters["model"]["output_size"],
            "number_of_connections": parameters["model"]["max_neighbors"],
        }

        self.output_size = parameters["model"]["output_size"]

        if model_type == "pn2":
            self.model = SegmentationPointNetP2(**model_kwargs)
        elif model_type == "pn":
            self.model = PointNetSegmenter(**model_kwargs)
        else:
            raise ValueError(f"Model type {model_type} not supported.")

        class_weights = torch.tensor([1.0, 2.0], device=device)  # Poids pour chaque classe
        self.loss = torch.nn.NLLLoss(weight=class_weights)  # Utilisation de NLLLoss avec log_softmax

        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.warmup = warmup

        self.val_step_outputs = torch.empty(0, self.output_size, device=device)
        self.val_step_targets = torch.empty(0, device=device, dtype=torch.long)

        # Variables pour enregistrer les résultats moyens
        # Calcul des métriques IoU et DSC
        self.fold_iou_scores_per_class = [[] for _ in range(self.output_size)] if not hasattr(self, 'fold_iou_scores_per_class') else self.fold_iou_scores_per_class
        self.fold_dice_scores_per_class = [[] for _ in range(self.output_size)] if not hasattr(self, 'fold_dice_scores_per_class') else self.fold_dice_scores_per_class
        self.fold_f1_scores = []

    def forward(self, graph: Batch):
        return self.model(graph)  # Sortie log_softmax

    def training_step(self, batch: Batch):
        pred = self.model(batch).to(torch.float32)  # log-probs
        target = batch.y.to(torch.long).to(pred.device)

        loss = self.loss(pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        with torch.no_grad():
            pred = self.model(batch).to(torch.float32)  # log-probs
        target = batch.y.to(torch.long).to(pred.device)

        self.val_step_outputs = torch.cat((self.val_step_outputs, pred), dim=0)
        self.val_step_targets = torch.cat((self.val_step_targets, target), dim=0)

        val_loss = self.loss(pred, target)
        self.log("validation_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Convert outputs to CPU numpy arrays
        predicteds = self.val_step_outputs.cpu()
        targets = self.val_step_targets.cpu()

        # Convertir log-probs → probs
        predicteds = torch.exp(predicteds)

        # Get predicted class indices
        predicted_classes = torch.argmax(predicteds, dim=1)

        # Histograms
        predicteds_np = predicteds.numpy()
        targets_np = targets.numpy()
        predicted_classes_np = predicted_classes.numpy()

        for i in range(self.output_size):
            true_mask = targets_np == i
            pred_mask = predicted_classes_np == i

            intersection = np.sum(true_mask & pred_mask)
            union = np.sum(true_mask | pred_mask)
            iou = intersection / union if union > 0 else 0.0
            dice = 2 * intersection / (np.sum(true_mask) + np.sum(pred_mask)) if (np.sum(true_mask) + np.sum(pred_mask)) > 0 else 0.0

            # Log par classe
            wandb.log({
            f"IoU/Class_{i}": iou,
            f"DSC/Class_{i}": dice
            })

            # Enregistrement des scores pour chaque classe
            self.fold_iou_scores_per_class[i].append(iou)
            self.fold_dice_scores_per_class[i].append(dice)

        # Compute F1 score
        f1 = f1_score(targets_np, predicted_classes_np, average="weighted")
        self.fold_f1_scores.append(f1)

        # Confusion matrix
        all_class_indices = torch.cat([predicted_classes, targets]).unique().cpu().numpy()
        all_class_indices = np.sort(all_class_indices)
        class_names = [f"Class {int(i)}" for i in all_class_indices]

        filtered_preds = predicted_classes_np[np.isin(predicted_classes_np, all_class_indices)]
        filtered_targets = targets_np[np.isin(targets_np, all_class_indices)]

        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            y_true=filtered_targets,
            preds=filtered_preds,
            class_names=class_names
        )})

        self.log("val_f1_score", f1, on_step=False, on_epoch=True, prog_bar=True)

        # Reset for next epoch
        self.val_step_outputs = torch.empty(0, self.output_size, device=self.device)
        self.val_step_targets = torch.empty(0, dtype=torch.long, device=self.device)

    def configure_optimizers(self):
        """Initialize the optimizer"""
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0001,
            betas=(0.9, 0.95),
        )
        sch = CosineWarmupScheduler(opt, warmup=self.warmup, max_iters=self.num_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
                "interval": "step",
                "frequency": 1,
            },
        }

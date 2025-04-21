import lightning as L
import torch
from segmentation_latents.model import SegmentationPointNetP2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import wandb
from torch_geometric.data import Batch
from segmentation_latents.utils.scheduler import CosineWarmupScheduler
from sklearn.utils import resample

class LightningModuleClassification(L.LightningModule):
    def __init__(
        self,
        parameters: dict,
        learning_rate: float,
        num_steps: int,
        warmup: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.param = parameters

        model_type = parameters["model"]["type"]

        model_kwargs ={
            "node_input_size": parameters["model"]["node_input_size"],
            "dim_model": parameters["model"]["dim_model"],
            "output_size": parameters["model"]["output_size"],
            "number_of_connections": parameters["dataset"]["number_of_connections"]
        }

        self.output_size = parameters["model"]["output_size"]

        if model_type == "pn2":
            self.model = SegmentationPointNetP2(**model_kwargs)
        elif model_type == "mlp":
            self.model = ClassificationModel(**model_kwargs)
        else:
            raise ValueError(f"Model type {model_type} not supported.")

        self.loss = torch.nn.NLLLoss()  # Utilisation de NLLLoss avec log_softmax

        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.warmup = warmup

        self.val_step_outputs = torch.empty(0, self.output_size, device=device)
        self.val_step_targets = torch.empty(0, device=device, dtype=torch.long)

    
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

        # Visualisation : on convertit en probabilités pour la lisibilité
        pred_probs = torch.exp(pred)
        positions = batch.pos.cpu().numpy()
        predicted_classes = torch.argmax(pred_probs, dim=1).cpu().numpy()

        for i in range(len(batch.ptr) - 1):
            start_idx, end_idx = batch.ptr[i].item(), batch.ptr[i + 1].item()
            graph_positions = positions[start_idx:end_idx]
            graph_predicted_classes = predicted_classes[start_idx:end_idx]

            views = [
                ("default", None, None),
                ("top", 90, 0),
                ("side", 0, 0),
                ("front", 0, 90),
            ]

            for view_name, elev, azim in views:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

                for class_id in np.unique(graph_predicted_classes):
                    class_points = graph_positions[graph_predicted_classes == class_id]
                    ax.scatter(class_points[:, 0], class_points[:, 1], class_points[:, 2],
                               s=1, label=f"Class {class_id}")

                ax.set_title(f"Segmentation Graph {i} - View: {view_name}")
                ax.set_xlabel("X Position")
                ax.set_ylabel("Y Position")
                ax.set_zlabel("Z Position")
                ax.legend()
                ax.set_box_aspect([1, 1, 1])

                if elev is not None and azim is not None:
                    ax.view_init(elev=elev, azim=azim)

                filename = f"segmentation_graph_{i}_{view_name}.png"
                plt.savefig(filename)
                wandb.log({f"Segmentation Graph {i} - {view_name}": wandb.Image(filename)})
                plt.close()

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

        # Calcul des métriques IoU et DSC
        iou_scores = []
        dice_scores = []
        for i in range(self.output_size):
            # On trouve les indices où les vraies classes sont égales à i
            true_class_mask = targets_np == i
            pred_class_mask = predicted_classes_np == i

            intersection = np.sum(true_class_mask & pred_class_mask)
            union = np.sum(true_class_mask | pred_class_mask)
            iou = intersection / union if union > 0 else 0.0
            iou_scores.append(iou)

            # Calcul du DSC
            dice = 2 * intersection / (np.sum(true_class_mask) + np.sum(pred_class_mask)) if (np.sum(true_class_mask) + np.sum(pred_class_mask)) > 0 else 0.0
            dice_scores.append(dice)

        # Calcul des intervalles de confiance CI 95% pour IoU et DSC
        def bootstrap_ci(scores, n_iterations=1000, ci_percentile=95):
            # Bootstrap pour calculer les intervalles de confiance
            resampled_scores = []
            for _ in range(n_iterations):
                resampled = resample(scores)
                resampled_scores.append(np.mean(resampled))
            lower = np.percentile(resampled_scores, (100 - ci_percentile) / 2)
            upper = np.percentile(resampled_scores, 100 - (100 - ci_percentile) / 2)
            return lower, upper

        iou_ci_lower, iou_ci_upper = bootstrap_ci(iou_scores)
        dice_ci_lower, dice_ci_upper = bootstrap_ci(dice_scores)

        # Log des métriques
        for i in range(self.output_size):
            wandb.log({
                f"IoU Class {i}": iou_scores[i],
                f"DSC Class {i}": dice_scores[i],
                f"IoU CI 95% Class {i}": f"({iou_ci_lower:.2f}, {iou_ci_upper:.2f})",
                f"DSC CI 95% Class {i}": f"({dice_ci_lower:.2f}, {dice_ci_upper:.2f})",
            })

        # Affichage des résultats dans le terminal
        print("IoU per class:", iou_scores)
        print("Dice per class:", dice_scores)
        print(f"IoU CI 95%: ({iou_ci_lower:.2f}, {iou_ci_upper:.2f})")
        print(f"Dice CI 95%: ({dice_ci_lower:.2f}, {dice_ci_upper:.2f})")

        # Confusion matrix et F1
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

        # Compute F1 score
        f1 = f1_score(targets_np, predicted_classes_np, average="weighted")
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

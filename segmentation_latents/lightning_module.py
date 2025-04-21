import lightning as L
import torch
from segmentation_latents.model import SegmentationPointNetP2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import wandb
from torch_geometric.data import Batch
from segmentation_latents.utils.scheduler import CosineWarmupScheduler

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

        if model_type =="pn2":
            self.model = SegmentationPointNetP2(**model_kwargs) 
        elif model_type == "mlp":
            self.model = ClassificationModel(**model_kwargs)
        else : 
            raise ValueError(f"Model type {model_type} not supported.")

        self.loss = torch.nn.NLLLoss() 

        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.warmup = warmup

        self.val_step_outputs = torch.empty(0, self.output_size, device=device)
        self.val_step_targets = torch.empty(0, device=device, dtype=torch.long)

    
    def forward(self, graph: Batch):
        return self.model(graph)

    def training_step(self, batch: Batch):
        pred = self.model(batch).to(torch.float32)
        target = batch.y.to(torch.long).to(pred.device) 

        loss = self.loss(pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):

        with torch.no_grad():
            pred = self.model(batch).to(torch.float32)
        target = batch.y.to(torch.long).to(pred.device)  

        self.val_step_outputs = torch.cat((self.val_step_outputs, pred), dim=0)
        self.val_step_targets = torch.cat((self.val_step_targets, target), dim=0)

        val_loss = self.loss(pred, target)

        self.log(
            "validation_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        # Get positions and predictions
        positions = batch.pos.cpu().numpy()
        predicted_classes = torch.argmax(pred, dim=1).cpu().numpy()

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
        # Convert outputs to numpy arrays
        predicteds = self.val_step_outputs.cpu().numpy()
        targets = self.val_step_targets.cpu().numpy()

        # Get the predicted class by taking the argmax of the softmax output
        predicted_classes = torch.argmax(torch.tensor(predicteds), axis=1)
        target_classes = targets


        # Histograms
        for i in range(self.output_size):
            plt.figure(figsize=(10, 6))
            for j in range(self.output_size):
                plt.hist(
                    predicteds[:, i][target_classes == j],
                    bins=40,
                    range=(0, 1),
                    alpha=0.7,
                    label=f"True Class {j}"
                )
            plt.ylabel("Frequency")
            plt.xlabel(f"Predicted Probabilities for Class {i}")
            plt.title(f"Validation Distribution - Class {i}")
            plt.legend(loc="upper right")

            filename = f"validation_histogram_plot_class_{i}.png"
            plt.savefig(filename)
            wandb.log({f"Histogram Class {i}": wandb.Image(filename)})
            plt.close()

        # Ensure both are tensors
        predicted_classes_tensor = torch.tensor(predicted_classes, device=self.device)
        target_classes_tensor = torch.tensor(target_classes, device=self.device)

        # Get all unique class indices from predictions and targets
        all_class_indices = torch.cat([predicted_classes_tensor, target_classes_tensor]).unique().cpu().numpy()
        class_names = [f"Class {i}" for i in all_class_indices]


        # Confusion Matrix
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            y_true=target_classes,
            preds=predicted_classes,
            class_names=class_names,
        )})


        f1 = f1_score(target_classes, predicted_classes, average="weighted")
        self.log("val_f1_score", f1, on_step=False, on_epoch=True, prog_bar=True)


        # Clear stored outputs
        self.val_step_outputs.clear()
        self.val_step_targets.clear()


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

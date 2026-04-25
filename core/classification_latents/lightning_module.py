import lightning as L
import torch
from .model import ClassificationModel, PointNetClassifier, ClassificationPointNetP2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import wandb
from torch_geometric.data import Batch
from .utils.scheduler import CosineWarmupScheduler

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
            "number_of_connections": parameters["dataset"]["number_of_connections"],
            "number_of_samples": parameters["dataset"]["number_of_samples"],
        }

        if model_type =="pn2":
            self.model = ClassificationPointNetP2(**model_kwargs) 
        elif model_type == "pn":
            self.model = PointNetClassifier(**model_kwargs)
        elif model_type == "mlp":
            self.model = ClassificationModel(**model_kwargs)
        else : 
            raise ValueError(f"Model type {model_type} not supported.")

        self.loss = torch.nn.BCELoss()

        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.warmup = warmup

        self.val_step_outputs = []
        self.val_step_targets = []

        self.test_step_outputs = []
        self.test_step_targets = []

    
    def forward(self, graph: Batch):
        return self.model(graph)

    def training_step(self, batch: Batch):
        pred = self.model(batch).reshape(-1).to(torch.float32)
        target = batch.y.to(torch.float32)

        loss = self.loss(pred, target)
        self.log("Training Loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):

        with torch.no_grad():
            pred = self.model(batch).reshape(-1).to(torch.float32)
        target = batch.y.to(torch.float32)
        self.val_step_outputs.append(pred.cpu())
        self.val_step_targets.append(target.cpu())

        pred = pred.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)

        val_loss = self.loss(pred, target)

        self.log(
            "Validation Loss", val_loss, on_step=True, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch: Batch):

        with torch.no_grad():
            pred = self.model(batch).reshape(-1).to(torch.float32)
        target = batch.y.to(torch.float32)
        self.test_step_outputs.append(pred.cpu())
        self.test_step_targets.append(target.cpu())
        
        pred = pred.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)

        test_loss = self.loss(pred, target)
        self.log("Test Loss", test_loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Convert outputs to numpy arrays
        predicteds = np.array(self.val_step_outputs)
        targets = np.array(self.val_step_targets)

        # Get the predicted class by taking the argmax of the softmax output
        predicted_classes = np.argmax(predicteds, axis=1)
        target_classes = np.argmax(targets, axis=1)

        # Create a histogram with two colors for each class
        plt.figure(figsize=(10, 6))
        plt.hist(
            [
                predicteds[:, 0][targets[:, 0] == 0],
                predicteds[:, 0][targets[:, 0] == 1],
            ],
            bins=20,
            range=(0, 1),
            color=["blue", "orange"],
            alpha=0.7,
            label=["Class 0", "Class 1"],
        )
        plt.ylabel("Frequency")
        plt.xlabel("Predicted Values")
        plt.title("Validation Predictions Distribution")
        plt.legend(loc="upper right")

        # Save the plot as an image
        plt.savefig("validation_histogram_plot.png")

        # Log the image to wandb
        wandb.log(
            {"Validation Histogram Plot": wandb.Image("validation_histogram_plot.png")}
        )

        # Close the plot to avoid memory issues
        plt.close()

        # Compute confusion matrix and F1 score
        conf_matrix = confusion_matrix(target_classes, predicted_classes, labels=[0, 1])
        f1 = f1_score(target_classes, predicted_classes, average="weighted")

        # Log la confusion matrix dans wandb
        wandb.log({"Validation Confusion Matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=target_classes,
            preds=predicted_classes,
            class_names=["Class 0", "Class 1"]
        )})
        self.log("F1 Score Validation", f1, on_step=False, on_epoch=True, prog_bar=True)

        # Compute accuracy for each class
        class_0_mask = target_classes == 0
        class_1_mask = target_classes == 1

        accuracy_class_0 = (predicted_classes[class_0_mask] == 0).sum() / class_0_mask.sum() if class_0_mask.sum() > 0 else 0.0
        accuracy_class_1 = (predicted_classes[class_1_mask] == 1).sum() / class_1_mask.sum() if class_1_mask.sum() > 0 else 0.0

        self.log("Accuracy Class 0", accuracy_class_0, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Accuracy Class 1", accuracy_class_1, on_step=False, on_epoch=True, prog_bar=True)

        # Clear stored outputs
        self.val_step_outputs.clear()
        self.val_step_targets.clear()

    def on_test_epoch_end(self):
        # Convert outputs to numpy arrays
        predicteds = np.array(self.test_step_outputs)
        targets = np.array(self.test_step_targets)

        # Get the predicted class by taking the argmax of the softmax output
        predicted_classes = np.argmax(predicteds, axis=1)
        target_classes = np.argmax(targets, axis=1)

        # Create a histogram with two colors for each class
        plt.figure(figsize=(10, 6))
        plt.hist(
            [
                predicteds[:, 0][targets[:, 0] == 0],
                predicteds[:, 0][targets[:, 0] == 1],
            ],
            bins=20,
            range=(0, 1),
            color=["blue", "orange"],
            alpha=0.7,
            label=["Class 0", "Class 1"],
        )
        plt.ylabel("Frequency")
        plt.xlabel("Predicted Values")
        plt.title("Test Predictions Distribution")
        plt.legend(loc="upper right")

        # Save the plot as an image
        plt.savefig("test_histogram_plot.png")

        # Log the image to wandb
        wandb.log(
            {"Test Histogram Plot": wandb.Image("test_histogram_plot.png")}
        )

        # Close the plot to avoid memory issues
        plt.close()

        # Compute confusion matrix and F1 score
        conf_matrix = confusion_matrix(target_classes, predicted_classes, labels=[0, 1])
        f1 = f1_score(target_classes, predicted_classes, average="weighted")

        self.log_dict(
            {
            "Aneurysm True pred": conf_matrix[0][0],
            "Aneurysm True target": 4,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            {
            "Vessel False pred": conf_matrix[0][1],
            "Vessel False Target": 0,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("F1 Score Test", f1, on_step=False, on_epoch=True, prog_bar=True)

        # Clear stored outputs
        self.test_step_outputs.clear()
        self.test_step_targets.clear()

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

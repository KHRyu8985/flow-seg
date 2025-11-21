import torch
import torch.nn as nn

class FocalL1Loss(nn.Module):
    """Focal L1 loss for medical image segmentation."""
    
    def __init__(
        self, 
        loss_type: int = 1, 
        alpha: float = 2.0, 
        beta: float = 1.0, 
        secondary_weight: float = 1.0, 
        pos_weight: float = 1.0, 
        base_loss: str = "smoothl1"
    ):
        super().__init__()
        self.base_loss = base_loss

        if base_loss == "smoothl1":
            self.criterion = nn.SmoothL1Loss(reduction="none")
        elif base_loss == "l2":
            self.criterion = nn.MSELoss(reduction="none")
        else:
            self.criterion = nn.L1Loss(reduction="none")

        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight
        self.secondary_weight = secondary_weight

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Forward pass of focal loss."""
        same_sign_mask = (torch.sign(pred) * torch.sign(gt) > 0)
        pos_mask = (gt > 0)
        weight = torch.ones_like(gt)

        weight = torch.where(
            same_sign_mask,
            torch.pow(torch.abs(pred - gt), self.beta) * self.alpha,
            torch.ones_like(gt),
        )
        weight = weight * torch.where(
            pos_mask,
            torch.full_like(gt, fill_value=self.pos_weight),
            torch.ones_like(gt),
        )

        if self.loss_type > 1:
            raise ValueError("Invalid type!")
        if self.loss_type in [0]:
            weight = weight.detach()

        loss = self.criterion(pred, gt)
        loss = loss * weight
        loss = torch.mean(loss, dim=(0, 2, 3))

        class_weights = torch.tensor(
            [1.0] + [self.secondary_weight] * (loss.shape[0] - 1), device=loss.device)
        loss = (loss * class_weights).sum() / class_weights.sum()
        return loss

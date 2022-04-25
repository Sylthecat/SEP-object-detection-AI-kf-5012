"""calculate loss for model"""
import torch
import torch.nn as nn
from SEP_AI_utils import intersection_over_union


class Loss(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        # S = split size of image,
        # B = number of boxes,
        # C = number of classes,

        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, preds, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5)
        preds = preds.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate intersection over union for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(preds[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(preds[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with the biggest intersection over union out of the two predictions
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)  # identity of obj_i

        # BOX COORDINATES

        # make boxes with no object in them = 0. We only take out one of the two
        # predictions, which is the one with biggest Iou calculated previously.
        box_predictions = exists_box * (
            (
                    bestbox * preds[..., 26:30]
                    + (1 - bestbox) * preds[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
            # 1e-6 gives numerical stability since the sqrt as you get
            # close to zero will be infinite
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        # (N, S, S, 4) - (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # OBJECT LOSS

        # pred_box is the confidence score for the bounding box with the biggest intersection over union
        pred_box = (
                bestbox * preds[..., 25:26] + (1 - bestbox) * preds[..., 20:21]
        )
        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # FOR NO OBJECT LOSS
        max_no_obj = torch.max(preds[..., 20:21], preds[..., 25:26])
        no_object_loss = self.mse(
           torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
           torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )
        # no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * preds[..., 20:21], start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        # )

        # no_object_loss += self.mse(
        #    torch.flatten((1 - exists_box) * preds[..., 25:26], start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        # )

        # FOR CLASS LOSS
        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * preds[..., :20], end_dim=-2, ),
            torch.flatten(exists_box * target[..., :20], end_dim=-2, ),
        )

        loss = (
                self.lambda_coord * box_loss  # first two rows of loss
                + object_loss  # third row of loss
                + self.lambda_noobj * no_object_loss  # forth row of loss
                + class_loss  # fifth row of loss
        )

        return loss

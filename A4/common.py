"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        dummy_out_shapes = {key: value.shape for key, value in dummy_out.items()}
        c3_channels = dummy_out_shapes["c3"][1]
        c4_channels = dummy_out_shapes["c4"][1]
        c5_channels = dummy_out_shapes["c5"][1]

        # lateral 1x1 conv layers
        self.fpn_params["lat_c3"] = nn.Conv2d(c3_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.fpn_params["lat_c4"] = nn.Conv2d(c4_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.fpn_params["lat_c5"] = nn.Conv2d(c5_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # output 3x3 conv layers (padding=1 保证尺寸不变)
        self.fpn_params["out_p3"] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.fpn_params["out_p4"] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.fpn_params["out_p5"] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code
        # 对每个尺度的特征应用 lateral 1x1 卷积，统一通道数
        lat_c5 = self.fpn_params["lat_c5"](backbone_feats["c5"])
        lat_c4 = self.fpn_params["lat_c4"](backbone_feats["c4"])
        lat_c3 = self.fpn_params["lat_c3"](backbone_feats["c3"])
        
        # 构建自顶向下的 FPN 特征
        # p5 直接使用 c5 的 lateral 特征
        p5 = lat_c5
        # 将 p5 上采样到 c4 的空间尺寸后与 c4 的 lateral 特征相加
        p4 = lat_c4 + torch.nn.functional.interpolate(p5, size=lat_c4.shape[-2:], mode='nearest')
        # 将 p4 上采样到 c3 的空间尺寸后与 c3 的 lateral 特征相加
        p3 = lat_c3 + torch.nn.functional.interpolate(p4, size=lat_c3.shape[-2:], mode='nearest')
        
        # 分别通过 3x3 卷积生成最终的 FPN 特征
        fpn_feats = {
            "p5": self.fpn_params["out_p5"](p5),
            "p4": self.fpn_params["out_p4"](p4),
            "p3": self.fpn_params["out_p3"](p3),
        }
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code
        H, W = feat_shape[2], feat_shape[3]  # Extract height and width
        stride = strides_per_fpn_level[level_name]  # Get stride

        # Generate grid of coordinates (centered in each cell)
        y, x = torch.meshgrid(
            torch.arange(H, dtype=dtype, device=device),
            torch.arange(W, dtype=dtype, device=device),
            indexing="ij",
        )

        # Convert to absolute coordinates in input image
        xc = (x + 0.5) * stride  # Center of receptive field
        yc = (y + 0.5) * stride

        # Stack and reshape to (H*W, 2)
        coords = torch.stack((xc, yc), dim=-1).reshape(-1, 2)

        location_coords[level_name] = coords
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code
    # Get indices of boxes sorted by scores in descending order
    def iou_between_boxes(a: torch.Tensor, b: torch.Tensor):
        # REMINDER: bot > top in img coordinates -- (0,0) starts @ top-left
        (ax1, ay1, ax2, ay2) = a[:,0], a[:,1], a[:,2], a[:,3]
        (bx1, by1, bx2, by2) = b[:,0], b[:,1], b[:,2], b[:,3]
        
        inter_left  = torch.maximum(ax1, bx1)
        inter_right = torch.minimum(ax2, bx2)
        inter_top   = torch.maximum(ay1, by1)
        inter_bot   = torch.minimum(ay2, by2)

        intersection = torch.clamp(inter_right - inter_left, 0) * torch.clamp(inter_bot - inter_top, 0)
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection

        return intersection / union
    
    # (1) (prepending '~' inverts the 'eliminated' mask)
    eliminated = torch.zeros(scores.shape, dtype=torch.bool, device=scores.device)
    _, sort_idx = torch.sort(scores, descending=True)

    for i, max_idx in enumerate(sort_idx): # 'i' being the index of sort_idx
        # if box was eliminated already
        if i + 1 >= len(sort_idx): break
        if eliminated[i]:
            continue
        
        box_max = boxes[max_idx]
        ious = iou_between_boxes(box_max.unsqueeze(0), boxes[sort_idx[i+1:]])

        elim_idx = torch.where(ious > iou_threshold)[0]
        if elim_idx.numel() != 0: # if no eliminations, do nothing
            elim_idx += i + 1
    
        eliminated[elim_idx] = True # (2)
        # print(eliminated)

        # --- sequential:
        # for box_idx in sort_idx[(i+1):]:
        #     if eliminated[box_idx]:
        #         continue
        #     box = boxes[box_idx]
        #     iou = iou_between_boxes(box_max, box.unsqueeze(0))
        #     print(iou)
        #     if iou > iou_threshold: # (2)
        #         eliminated[box_idx] = True

    keep = sort_idx[~eliminated]
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep

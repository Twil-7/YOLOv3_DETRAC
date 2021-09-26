from keras import backend as k
import numpy as np


with open("yolo_anchors.txt", "r") as f:
    string = f.read().strip().split(',')
anchors = [int(s) for s in string]
anchors = np.reshape(anchors, (9, 2))


def yolo_head(y_pre_part):

    # y_pre_part : Tensor, shape=(batch, 13, 13, 75), float32

    grid_shape = k.shape(y_pre_part)[1:3]
    # Tensor, shape=(2, ), [13 13]

    grid_y = k.tile(k.reshape(k.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    # Tensor, shape=(13, 13, 1, 1), int32
    grid_x = k.tile(k.reshape(k.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    # Tensor, shape=(13, 13, 1, 1), int32
    grid = k.concatenate([grid_x, grid_y])
    # Tensor, shape=(13, 13, 1, 2), int32
    grid = k.cast(grid, k.dtype(y_pre_part))
    # Tensor, shape=(13, 13, 1, 2), float32

    return grid


def yolo_loss(args):

    y_pre = args[:3]
    y_true = args[3:]
    # y_pre :  [ Tensor (batch, 13, 13, 15) , Tensor (batch, 26, 26, 15) , Tensor (batch, 52, 52, 15) ]
    # y_true : [ Tensor (batch, 13, 13, 3, 5) , Tensor (batch, 26, 26, 3, 5) , Tensor (batch, 52, 52, 3, 5) ]

    # little grid predict large bounding box and
    # large grid predict little bounding box
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = k.shape(y_pre[0])[1:3] * 32
    input_shape = k.cast(input_shape, k.dtype(y_true[0]))
    # [13, 13] * 32 = [416, 416] - tensor, float

    grid_shapes = [k.cast(k.shape(y_pre[l])[1:3], k.dtype(y_true[0])) for l in range(3)]
    # [[13, 13]-tensor, [26, 26]-tensor, [52, 52]-tensor]

    loss = 0
    m = k.cast(k.shape(y_pre[0])[0], k.dtype(y_pre[0]))    # batch size, tensor-32, float

    for l in range(3):
        # y_pre[l] : shape=(batch, ?, ?, 15), float32
        # single_y_pre : shape=(batch, ?, ?, 3, 5), float32
        # grid : shape=(?, ?, 1, 2), float32

        grid_shape = k.shape(y_pre[l])[1:3]    # Tensor, shape=(2, ), [13, 13] or [26, 26] or [52, 52]
        single_y_pre = k.reshape(y_pre[l], [-1, grid_shape[0], grid_shape[1], 3, 5])
        # Tensor, shape=(batch, ?, ?, 3, 5), float32
        grid = yolo_head(y_pre[l])
        # Tensor, shape=(?, ?, 1, 2), float32

        # calculate the y_true_confidence, y_true_class, y_true_xy, y_true_wh---------------------------------------

        y_true_confidence = y_true[l][..., 4:5]    # Tensor, (batch, 13, 13, 3, 1), float32
        object_mask = y_true_confidence

        y_true_xy = y_true[l][..., 0:2]*grid_shapes[l][::-1] - grid    # Tensor, shape=(batch, 13, 13, 3, 2), float32

        y_true_wh = k.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        y_true_wh = k.switch(object_mask, y_true_wh, k.zeros_like(y_true_wh))
        # avoid log(0)=-inf
        # Tensor, shape=(batch, 13, 13, 3, 2), float32

        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]
        # Tensor, shape=(batch, 13, 13, 3, 1), float32

        # calculate the y_pre_confidence, y_pre_class, y_pre_xy, y_pre_wh-----------------------------------------

        y_pre_confidence = single_y_pre[..., 4:5]
        # Tensor, shape=(batch, 13, 13, 3, 1), float32
        y_pre_xy = single_y_pre[..., 0:2]
        # Tensor, shape=(batch, 13, 13, 3, 2), float32
        y_pre_wh = single_y_pre[..., 2:4]
        # Tensor, shape=(batch, 13, 13, 3, 2), float32

        # calculate the sum loss ---------------------------------------------------------------------------------

        xy_loss = object_mask * box_loss_scale * k.binary_crossentropy(y_true_xy, y_pre_xy, from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * k.square(y_true_wh - y_pre_wh)

        confidence_loss1 = object_mask * k.binary_crossentropy(y_true_confidence, y_pre_confidence, from_logits=True)
        confidence_loss2 = (1-object_mask) * k.binary_crossentropy(y_true_confidence, y_pre_confidence, from_logits=True)
        confidence_loss = confidence_loss1 + confidence_loss2

        xy_loss = k.sum(xy_loss) / m
        wh_loss = k.sum(wh_loss) / m
        confidence_loss = k.sum(confidence_loss) / m
        loss += xy_loss + wh_loss + confidence_loss

    return loss

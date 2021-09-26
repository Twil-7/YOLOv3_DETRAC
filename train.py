import numpy as np
from keras.models import load_model
import cv2
import yolov3_model
from yolov3_loss import yolo_loss
from keras.utils import Sequence
import math
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import Adam


with open("yolo_anchors.txt", "r") as f:
    string = f.read().strip().split(',')
anchors = [int(s) for s in string]
anchors = np.reshape(anchors, (9, 2))

# [[22  37]
#  [26  82]
#  [49  132]
#  [56  57]
#  [89  211]
#  [113 108]
#  [162 298]
#  [238 177]
#  [341 340]]


def max_iou_index(w, h):

    # (w, h) : 253 177
    box_area = np.array([w * h]).repeat(9)
    # box_area : [44781 44781 44781 44781 44781 44781 44781 44781 44781]
    anchor_area = anchors[:, 0] * anchors[:, 1]
    # anchor_area : [875 2618 7168 3588 19600 16688 47435 51510 117978]

    anchor_w_matrix = anchors[:, 0]
    # anchor_w_matrix : [25 34 64 69 100 149 179 303 371]
    min_w_matrix = np.minimum(anchor_w_matrix, w)
    # min_w_matrix : [25 34 64 69 100 149 179 253 253]

    anchor_h_matrix = anchors[:, 1]
    # anchor_h_matrix : [35 77 112 52 196 112 265 170 318]
    min_h_matrix = np.minimum(anchor_h_matrix, h)
    # min_h_matrix : [35 77 112 52 177 112 177 170 177]

    inter_area = np.multiply(min_w_matrix, min_h_matrix)
    # inter_area : [875 2618 7168 3588 17700 16688 31683 43010 44781]
    iou = inter_area / (box_area + anchor_area - inter_area)
    # iou : [0.01953954 0.05846229 0.16006789 0.08012327 0.37916926 0.37265805 0.52340046 0.80722959 0.37957077]
    index = np.argmax(iou)

    return index


def data_encoding(batch_image_path, batch_true_boxes):

    # batch_image_path : (32, str)， str recorded the image path
    # batch_true_boxes : (32, str)， str has absolute x_min, y_min, x_max, y_max, class_id
    # encoding_y : x、y、w、h are relative value

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    grid_shapes = [[13, 13], [26, 26], [52, 52]]
    batch_encoding_y = [np.zeros((32, 13, 13, 3, 5)), np.zeros((32, 26, 26, 3, 5)), np.zeros((32, 52, 52, 3, 5))]
    batch_images = []

    for i in range(len(batch_true_boxes)):

        img = cv2.imread(batch_image_path[i])
        size = img.shape

        img1 = img / 255
        resize_img = cv2.resize(img1, (416, 416), interpolation=cv2.INTER_AREA)
        batch_images.append(resize_img)

        obj1 = batch_true_boxes[i].strip().split(' ')
        for j in range(len(obj1)):

            obj2 = obj1[j].split('_')

            x1 = int(obj2[0])
            y1 = int(obj2[1])
            x2 = int(obj2[2])
            y2 = int(obj2[3])

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            center_x_ratio = center_x / size[1]
            center_y_ratio = center_y / size[0]
            w_ratio = w / size[1]
            h_ratio = h / size[0]

            anchor_index = max_iou_index(w, h)

            for num in range(3):
                if anchor_index in anchor_mask[num]:

                    inner_index = anchor_mask[num].index(anchor_index)   # 0 or 1 or 2

                    grid_x = int(center_x / size[1] * grid_shapes[num][1])
                    grid_y = int(center_y / size[0] * grid_shapes[num][0])

                    batch_encoding_y[num][i, grid_y, grid_x, inner_index, 0:4] = np.array([center_x_ratio,
                                                                                           center_y_ratio,
                                                                                           w_ratio, h_ratio])
                    batch_encoding_y[num][i, grid_y, grid_x, inner_index, 4] = 1

    return batch_images, batch_encoding_y


class SequenceData(Sequence):

    def __init__(self, data_x, data_y, batch_size):
        self.batch_size = batch_size
        self.data_x = data_x
        self.data_y = data_y
        self.indexes = np.arange(len(self.data_x))

    def __len__(self):
        return math.floor(len(self.data_x) / float(self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, idx):

        batch_index = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.data_x[k] for k in batch_index]
        batch_y = [self.data_y[k] for k in batch_index]

        x, y = data_encoding(batch_x, batch_y)
        x = np.array(x)

        return [x, *y], np.zeros(32)


# create model and train and save
def train_network(train_generator, validation_generator, epoch):

    model_body = yolov3_model.create_yolo_model()
    model_body.load_weights('/home/zk/Desktop/YOLOv3_Car_07_12/yolo_weights.h5', by_name=True, skip_mismatch=True)

    print('model_body layers : ', len(model_body.layers))
    for i in range(249):
        model_body.layers[i].trainable = True
    print('249 Layers has been frozen ! ')

    grid_shape = np.array([[13, 13], [26, 26], [52, 52]])

    y_true = [Input(shape=(grid_shape[l, 0], grid_shape[l, 1], 3, 5)) for l in range(3)]
    # [Tensor-(?, 13, 13, 3, 25), Tensor-(?, 26, 26, 3, 25), Tensor-(?, 52, 52, 3, 25)]

    model_loss = Lambda(yolo_loss, output_shape=(1, ), name='yolo_loss')([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    # model_body.input : Tensor, shape=(?, 416, 416, 3), float32

    model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pre: y_pre})

    checkpoint = ModelCheckpoint('/home/zk/Desktop/YOLOv3_Car_07_12/best_weights.hdf5', monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model_body.save_weights('first_weights.hdf5')


def load_network_then_train(train_generator, validation_generator, epoch, input_name, output_name):

    model_body = yolov3_model.create_yolo_model()
    model_body.load_weights(input_name, by_name=True, skip_mismatch=True)

    for i in range(249):
        model_body.layers[i].trainable = True

    grid_shape = np.array([[13, 13], [26, 26], [52, 52]])
    y_true = [Input(shape=(grid_shape[l, 0], grid_shape[l, 1], 3, 5)) for l in range(3)]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss')([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    sgd = optimizers.SGD(lr=1e-4, momentum=0.9)
    model.compile(optimizer=sgd, loss={'yolo_loss': lambda y_true, y_pre: y_pre})

    checkpoint = ModelCheckpoint('/home/zk/Desktop/YOLOv3_Car_07_12/best_weights.hdf5', monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights(output_name)

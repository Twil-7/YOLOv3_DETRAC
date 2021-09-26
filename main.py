import numpy as np
import cv2
import read_data_path as rp
import get_anchors as ga
import yolov3_model as ym
import train as tr
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def txt_document(matrix):
    f = open("yolo_anchors.txt", 'w')
    row = np.shape(matrix)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (matrix[i][0], matrix[i][1])
        else:
            x_y = ", %d,%d" % (matrix[i][0], matrix[i][1])
        f.write(x_y)

    f.close()


def sigmoid(x):
    return 1/(1 + np.exp(-x))


if __name__ == "__main__":

    train_x1, train_y1, test_x1, test_y1 = rp.make_data()

    # for i in range(1000, 1100):
    #
    #     img1 = cv2.imread(train_x[i])
    #     size = img1.shape
    #     img = cv2.resize(img1, (416, 416), interpolation=cv2.INTER_AREA)
    #
    #     obj1 = train_y[i].strip().split(' ')
    #     for j in range(len(obj1)):
    #
    #         obj2 = obj1[j].split('_')
    #
    #         x1 = int(int(obj2[0])/size[1]*416)
    #         y1 = int(int(obj2[1])/size[0]*416)
    #         x2 = int(int(obj2[2])/size[1]*416)
    #         y2 = int(int(obj2[3])/size[0]*416)
    #
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    #     cv2.namedWindow("Image")
    #     cv2.imshow("Image", img)
    #     cv2.waitKey(0)

    anchors = ga.calculate_anchor(train_y1)

    print(anchors)

    # [[4 11]
    #  [6 16]
    #  [9 23]
    #  [13 29]
    #  [16 43]
    #  [20 30]
    #  [22 66]
    #  [28 45]
    #  [40 84]]

    txt_document(anchors)
    print('The anchors have been documented ! ')

    # train_x = train_x1[:10000]
    # train_y = train_y1[:10000]
    # test_x = test_x1[:5000]
    # test_y = test_y1[:5000]

    train_generator = tr.SequenceData(train_x1, train_y1, 32)
    test_generator = tr.SequenceData(test_x1, test_y1, 32)

    # tr.train_network(train_generator, test_generator, epoch=10)
    # tr.load_network_then_train(train_generator, test_generator, epoch=3,
    #                            input_name='train_2_epoch_weights.hdf5', output_name='first_weights.hdf5')

    yolo3_model = ym.create_yolo_model()

    yolo3_model.load_weights('best_weights.hdf5')

    for i in range(1000):

        img1 = cv2.imread(test_x1[i])
        size = img1.shape

        img2 = img1 / 255
        img3 = cv2.resize(img2, (416, 416), interpolation=cv2.INTER_AREA)
        img4 = img3[np.newaxis, :, :, :]

        pre = yolo3_model.predict(img4)

        pre_13 = np.reshape(pre[0][0], [13, 13, 3, 5])
        pre_26 = np.reshape(pre[1][0], [26, 26, 3, 5])
        pre_52 = np.reshape(pre[2][0], [52, 52, 3, 5])

        grid_y_13 = np.tile(np.reshape(np.arange(0, 13), [-1, 1]), [1, 13]) * 416 / 13
        # 0   0   0   ...
        # 32  32  32  ...
        # ... ...     ...
        # 384 384 384 ...

        grid_x_13 = np.tile(np.reshape(np.arange(0, 13), [1, -1]), [13, 1]) * 416 / 13
        # 0  32  64 ...
        # 0  32  64 ...
        # ...  ...  ...
        # 0  32  64 ...

        grid_y_26 = np.tile(np.reshape(np.arange(0, 26), [-1, 1]), [1, 26]) * 416 / 26  # (26, 26)
        grid_x_26 = np.tile(np.reshape(np.arange(0, 26), [1, -1]), [26, 1]) * 416 / 26  # (26, 26)
        grid_y_52 = np.tile(np.reshape(np.arange(0, 52), [-1, 1]), [1, 52]) * 416 / 52  # (52, 52)
        grid_x_52 = np.tile(np.reshape(np.arange(0, 52), [1, -1]), [52, 1]) * 416 / 52  # (52, 52)

        for j in range(3):
            pre_13[:, :, j, 0] = sigmoid(pre_13[:, :, j, 0]) * 416 / 13 + grid_x_13
            pre_13[:, :, j, 1] = sigmoid(pre_13[:, :, j, 1]) * 416 / 13 + grid_y_13
            pre_13[:, :, j, 2] = np.exp(pre_13[:, :, j, 2]) * anchors[j + 6, 0]
            pre_13[:, :, j, 3] = np.exp(pre_13[:, :, j, 3]) * anchors[j + 6, 1]
            pre_13[:, :, j, 4] = sigmoid(pre_13[:, :, j, 4])

            pre_26[:, :, j, 0] = sigmoid(pre_26[:, :, j, 0]) * 416 / 26 + grid_x_26
            pre_26[:, :, j, 1] = sigmoid(pre_26[:, :, j, 1]) * 416 / 26 + grid_y_26
            pre_26[:, :, j, 2] = np.exp(pre_26[:, :, j, 2]) * anchors[j + 3, 0]
            pre_26[:, :, j, 3] = np.exp(pre_26[:, :, j, 3]) * anchors[j + 3, 1]
            pre_26[:, :, j, 4] = sigmoid(pre_26[:, :, j, 4])

            pre_52[:, :, j, 0] = sigmoid(pre_52[:, :, j, 0]) * 416 / 52 + grid_x_52
            pre_52[:, :, j, 1] = sigmoid(pre_52[:, :, j, 1]) * 416 / 52 + grid_y_52
            pre_52[:, :, j, 2] = np.exp(pre_52[:, :, j, 2]) * anchors[j, 0]
            pre_52[:, :, j, 3] = np.exp(pre_52[:, :, j, 3]) * anchors[j, 1]
            pre_52[:, :, j, 4] = sigmoid(pre_52[:, :, j, 4])

        candidate_box = []

        for k1 in range(13):
            for k2 in range(13):
                for k3 in range(3):

                    if pre_13[k1, k2, k3, 4] > 0.5:
                        center_x = pre_13[k1, k2, k3, 0]
                        center_y = pre_13[k1, k2, k3, 1]
                        w = pre_13[k1, k2, k3, 2]
                        h = pre_13[k1, k2, k3, 3]
                        confidence = pre_13[k1, k2, k3, 4]

                        x1 = center_x - w / 2
                        y1 = center_y - h / 2
                        x2 = center_x + w / 2
                        y2 = center_y + h / 2

                        candidate_box.append([x1, y1, x2, y2, confidence])
                        print('Grid 13 * 13 :', x1, y1, x2, y2, confidence)

        for k1 in range(26):
            for k2 in range(26):
                for k3 in range(3):

                    if pre_26[k1, k2, k3, 4] > 0.5:
                        center_x = pre_26[k1, k2, k3, 0]
                        center_y = pre_26[k1, k2, k3, 1]
                        w = pre_26[k1, k2, k3, 2]
                        h = pre_26[k1, k2, k3, 3]
                        confidence = pre_26[k1, k2, k3, 4]

                        x1 = center_x - w / 2
                        y1 = center_y - h / 2
                        x2 = center_x + w / 2
                        y2 = center_y + h / 2

                        candidate_box.append([x1, y1, x2, y2, confidence])
                        print('Grid 26 * 26 :', x1, y1, x2, y2, confidence)

        for k1 in range(52):
            for k2 in range(52):
                for k3 in range(3):

                    if pre_52[k1, k2, k3, 4] > 0.5:
                        center_x = pre_52[k1, k2, k3, 0]
                        center_y = pre_52[k1, k2, k3, 1]
                        w = pre_52[k1, k2, k3, 2]
                        h = pre_52[k1, k2, k3, 3]
                        confidence = pre_52[k1, k2, k3, 4]

                        x1 = center_x - w / 2
                        y1 = center_y - h / 2
                        x2 = center_x + w / 2
                        y2 = center_y + h / 2

                        candidate_box.append([x1, y1, x2, y2, confidence])
                        print('Grid 52 * 52 :', x1, y1, x2, y2, confidence)

        candidate_box = np.array(candidate_box)

        for num in range(len(candidate_box)):
            a1 = int(candidate_box[num, 0] / 416 * size[1])
            b1 = int(candidate_box[num, 1] / 416 * size[0])
            a2 = int(candidate_box[num, 2] / 416 * size[1])
            b2 = int(candidate_box[num, 3] / 416 * size[0])
            confidence = str(candidate_box[num, 4])

            cv2.rectangle(img1, (a1, b1), (a2, b2), (0, 0, 255), 2)
            cv2.putText(img1, confidence, (a2, int((b1 + b2) / 2)), 1, 1, (0, 0, 255))

        # cv2.namedWindow("Final_Image")
        # cv2.imshow("Final_Image", img1)
        # cv2.waitKey(0)

        cv2.imwrite("/home/zk/Desktop/YOLOv3_Car_07_12/demo/" + str(i) + '.jpg', img1)



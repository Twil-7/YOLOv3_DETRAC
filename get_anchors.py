import numpy as np
import cv2


# the anchor_iou between 19670 box and 9 clusters
def anchor_iou(boxes, clusters):
    n = boxes.shape[0]    # 19670
    k = clusters.shape[0]    # 9

    box_area = boxes[:, 0] * boxes[:, 1]    # 19670
    # repeat function : [1 1 1 2 2 2 3 3 3]
    box_area = box_area.repeat(k)    # 19670 * 9 = 177030
    box_area = np.reshape(box_area, (n, k))    # (19670, 9), every column is the box_area vector

    cluster_area = clusters[:, 0] * clusters[:, 1]    # 9
    # tile function : [1 2 3 1 2 3 1 2 3]
    cluster_area = np.tile(cluster_area, [1, n])    # 9 * 19670 = 177030
    cluster_area = np.reshape(cluster_area, (n, k))    # (19670, 9), every row is the cluster_area vector

    box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))    # (19670, 9)
    cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))    # (19670, 9)
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)    # (19670, 9)

    box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))    # (19670, 9)
    cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))    # (19670, 9)
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)    # (19670, 9)

    inter_area = np.multiply(min_w_matrix, min_h_matrix)    # (19670, 9)
    iou = inter_area / (box_area + cluster_area - inter_area)    # (19670, 9)

    return iou


def k_means(box, k):
    box_number = len(box)    # 19670
    last_nearest = np.zeros(box_number)

    # init k clusters
    np.random.seed(1)
    clusters = box[np.random.choice(box_number, k, replace=False)]
    # (9, 2)

    while True:

        # calculate the iou_distance between 19670 boxes and 9 clusters
        # distance :  (19670, 9)
        distances = 1 - anchor_iou(box, clusters)

        # calculate the mum anchor index for 19670 boxes
        # current_nearest : (19670, 1)
        current_nearest = np.argmin(distances, axis=1)

        if (last_nearest == current_nearest).all():
            break

        for num in range(k):
            clusters[num] = np.median(box[current_nearest == num], axis=0)
        last_nearest = current_nearest

    return clusters


def calculate_anchor(train_y):
    box = []

    # len(train_y) : 19670
    for i in range(len(train_y)):

        size = [540, 960]
        obj1 = train_y[i].strip().split(' ')
        for j in range(len(obj1)):

            obj2 = obj1[j].split('_')

            x1 = int(int(obj2[0]) / size[1] * 416)
            y1 = int(int(obj2[1]) / size[0] * 416)
            x2 = int(int(obj2[2]) / size[1] * 416)
            y2 = int(int(obj2[3]) / size[0] * 416)

            w = x2 - x1
            h = y2 - y1

            w = int(w / size[1] * 416)
            h = int(h / size[0] * 416)

            box.append([w, h])

    box = np.array(box)
    print('All boxes : ', len(box))
    # box : (19670, 2)

    anchors = k_means(box, 9)
    # anchors : (9, 2)

    index = np.argsort(anchors[:, 0])
    # [0 5 6 8 2 7 4 3 1]

    anchors = anchors[index]

    # [[93  22]
    #  [117 35]
    #  [117 26]
    #  [139 50]
    #  [140 30]
    #  [151 35]
    #  [164 41]
    #  [177 35]
    #  [194 44]]

    return anchors




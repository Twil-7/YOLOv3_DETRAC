import os
import cv2
import xml.etree.ElementTree as ET


def read_data(path_xml, path_img):
    data_x = []
    data_y = []

    xml_path1 = path_xml
    img_path1 = path_img

    xml_filename1 = os.listdir(xml_path1)
    xml_filename1.sort()

    for name1 in xml_filename1:

        xml_path2 = xml_path1 + '/' + name1
        img_path2 = img_path1 + '/' + name1.split('.')[0]

        tree = ET.parse(xml_path2)
        root = tree.getroot()

        for xml_frame in root:

            if xml_frame.tag == "frame":

                img_path3 = img_path2 + '/img' + xml_frame.attrib["num"].zfill(5) + '.jpg'
                data_x.append(img_path3)

                coordinate_txt = ''
                xml_target_list = xml_frame[0]

                for xml_target in xml_target_list:

                    if xml_target.tag == "target":

                        for xml_child in xml_target:

                            if xml_child.tag == "box":
                                x_min = int(float(xml_child.attrib["left"]))
                                y_min = int(float(xml_child.attrib["top"]))
                                box_width = int(float(xml_child.attrib["width"]))
                                box_height = int(float(xml_child.attrib["height"]))

                                x1 = x_min
                                y1 = y_min
                                x2 = x_min + box_width
                                y2 = y_min + box_height

                                txt = str(x1) + '_' + str(y1) + '_' + str(x2) + '_' + str(y2) + " "
                                coordinate_txt = coordinate_txt + txt

                data_y.append(coordinate_txt)

    return data_x, data_y


def make_data():

    train_xml_path = '/home/zk/Desktop/YOLOv3_Car_07_12/DETRAC-Train-Annotations-XML'
    train_img_path = '/home/zk/Desktop/YOLOv3_Car_07_12/Insight-MVT_Annotation_Train'
    test_xml_path = '/home/zk/Desktop/YOLOv3_Car_07_12/DETRAC-Test-Annotations-XML'
    test_img_path = '/home/zk/Desktop/YOLOv3_Car_07_12/Insight-MVT_Annotation_Test'

    train_x, train_y = read_data(train_xml_path, train_img_path)
    print('train_x num : ', len(train_x), '    ', 'train_y num : ', len(train_y))
    test_x, test_y = read_data(test_xml_path, test_img_path)
    print('test_x num : ', len(test_x), '    ', 'test_y num : ', len(test_y))

    # train_x num: 82085
    # train_y num: 82085
    # test_x num : 56167
    # test_y num : 56167

    return train_x, train_y, test_x, test_y

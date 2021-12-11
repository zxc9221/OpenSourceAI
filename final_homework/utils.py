import cv2
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import random


def get_ious(pred_coords, true_coord):
    """
    Descriptions:
    복수개의 좌표들에 예측 좌표에 대한 하나의 true coord와 iou 을 계산합니다.
    :param pred_coords: ndarray, shape (N, 4)
    :param true_coords: ndarray, shape (1, 4)
    :return: ious: ndarray, shape (N, 4)
    """
    ious = []
    pred_coords = pred_coords.reshape(pred_coords.shape[1] * pred_coords.shape[2], 4)
    for coord in pred_coords:
        # convert (cx, cy, h, w) -> (x1, y1, x2, y2)
        pred_cx, pred_cy, pred_h, pred_w = coord
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_cx - pred_w / 2, pred_cy - pred_h / 2, pred_cx + pred_w / 2, pred_cy + pred_h / 2

        true_cx, true_cy, true_h, true_w = true_coord
        true_x1, true_y1, true_x2, true_y2 = true_cx - true_w / 2, true_cy - true_h / 2, true_cx + true_w / 2, true_cy + true_h / 2

        # calculate each box area
        pred_box_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        true_box_area = (true_x2 - true_x1) * (true_y2 - true_y1)

        # get coord of inter area
        inter_x1 = max(pred_x1, true_x1)
        inter_y1 = max(pred_y1, true_y1)
        inter_x2 = min(pred_x2, true_x2)
        inter_y2 = min(pred_y2, true_y2)

        # calculate inter box w, h
        inter_w = inter_x2 - inter_x1
        inter_h = inter_y2 - inter_y1

        # calculate inter box area
        inter_area = inter_w * inter_h
        iou = inter_area / (pred_box_area + true_box_area - inter_area)

        ious.append([iou])
    return ious


def original_rectangle_coords(fmap_size, kernel_sizes, strides, paddings):
    """
    Description:
    주어진 Feature map의 center x, center y 좌표를 Original Image center x, center y에 맵핑합니다.

    아래 코드에서 사용된 공식은 "A guide to convolution arithmetic for deep learning" 에서 가져옴


    :param fmap_size: 1d array, 최종 출력된 fmap shape, (H, W) 로 구성
        example) (4, 4)
    :param kernel_sizes: tuple or list, 각 Layer 에 적용된 filter 크기,
        example) [3, 3]
    :param strides: tuple or list, 각 Layer 에 적용된 stride 크기
        example) [2, 1]
    :param paddings: tuple or list, List 의 Element 가 'SAME', 'VALID' 로 구성되어 있어야 함
        example) ['SAME', 'VALID']
    :return:
    """

    rf = 1  # receptive field
    jump = 1  # 점과 점사이의 거리
    start_out = 0.5
    assert len(kernel_sizes) == len(strides) == len(paddings), 'kernel sizes, strides, paddings 의 크기가 같아야 합니다.'

    for stride, kernel_size, padding in zip(strides, kernel_sizes, paddings):
        # padding 의 크기를 계산합니다.
        if padding == 'SAME':
            padding = (kernel_size - 1) / 2
        else:
            padding = 0

        # 시작점을 계산합니다.
        start_out += ((kernel_size - 1) * 0.5 - padding) * jump

        # receptive field 을 계산합니다.
        rf += (kernel_size - 1) * jump

        # 점과 점사이의 거리를 계산합니다.
        jump *= stride

    xs, ys = np.meshgrid(range(fmap_size[0]), range(fmap_size[1]))
    xs = xs * jump + start_out
    ys = ys * jump + start_out
    ys = ys.ravel()
    xs = xs.ravel()
    n_samples = len(xs)

    # coords = ((cx, cy, w, h), (cx, cy, w, h) ... (cx, cy, w, h))
    coords = np.stack([ys, xs, [rf] * n_samples, [rf] * n_samples], axis=-1)
    return coords


def xyxy2xywh(xyxy):
    """
    Description:
    x1 y1 x2 y2 좌표를 cx cy w h 좌표로 변환합니다.

    :param xyxy: shape (..., 4), 2차원 이상의 array 가 들어와야 함
    :return: xywh shape(... , 4), 2차원 이상의 array 가 들어와야 함
    """
    w = xyxy[..., 2:3] - xyxy[..., 0:1]
    h = xyxy[..., 3:4] - xyxy[..., 1:2]
    cx = xyxy[..., 0:1] + w * 0.5
    cy = xyxy[..., 1:2] + h * 0.5
    xywh = np.concatenate([cx, cy, w, h], axis=-1)
    return xywh


def xywh2xyxy(xywh):
    """
    Description:
    cx cy w h 좌표를 x1 y1 x2 y2 좌표로 변환합니다.

    center x, center y, w, h 좌표계를 가진 ndarray 을 x1, y1, x2, y2 좌표계로 변경
    xywh : ndarary, shape, (..., 4), 마지막 차원만 4 이면 작동.
    """
    cx = xywh[..., 0]
    cy = xywh[..., 1]
    w = xywh[..., 2]
    h = xywh[..., 3]

    x1 = cx - w * 0.5
    x2 = cx + w * 0.5
    y1 = cy - h * 0.5
    y2 = cy + h * 0.5

    return np.stack([x1, y1, x2, y2], axis=-1)


def draw_rectangle(img, coordinate, color=(255, 0, 0)):
    """
    Description:
    img 에 하나의 bounding box 을 그리는 함수.

    :param img: ndarray 2d (gray img)or 3d array(color img),
    :param coordinate: tuple or list(iterable), shape=(4,) x1 ,y1, x2, y2
    :param color: tuple(iterable), shape = (3,)
    :return:
    """

    # opencv 에 입력값으로 넣기 위해 반드시 정수형으로 변경해야함
    coordinate = coordinate.astype('int64')
    x_min = coordinate[0]
    x_max = coordinate[2]
    y_min = coordinate[1]
    y_max = coordinate[3]

    img = img.astype('uint8')

    return cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color)


def draw_rectangles(img, coordinates):
    """
    Description:
    하나의 img 에 복수개의  bounding box 을 그리는 함수.

    :param img: ndarray 2d (gray img)or 3d array(color img),
    :param coordinates: tuple or list(iterable), ((x y, x, y), (x y, x, y) .. (x y, x, y))
    내부적으로는 x y x y 좌표계를 가지고 있어야 함
    :return: img: ndarray, 3d array, shape = (H, W, CH)
    """
    for coord in coordinates:
        img = draw_rectangle(img, coord)
    return np.array(img)


def images_with_rectangles(imgs, bboxes_bucket):
    """
    여러개의 이미지에 여러개의 bouding box 을 그리는 알고리즘.

    :param imgs: ndarray , 4d array, N H W C 구조를 가지고 있음
    :param bboxes_bucket:tuple or list(iterable),
        (
        (x y, x, y), (x y, x, y) .. (x y, x, y),
        (x y, x, y), (x y, x, y) .. (x y, x, y),
                    ...
        (x y, x, y), (x y, x, y) .. (x y, x, y),
        )
    :return: list, 4d array,  N H W C 구조를 가지고 있음
    """
    boxed_imgs = []
    for img, bboxes in zip(imgs, bboxes_bucket):
        # if gray image
        if img.shape[-1] == 1:
            img = np.squeeze(img)
        # draw bbox img
        bboxes_img = draw_rectangles(img, bboxes)
        boxed_imgs.append(bboxes_img)
    return boxed_imgs


def plot_images(imgs, names=None, random_order=False, savepath=None):
    h = math.ceil(math.sqrt(len(imgs)))
    fig = plt.figure()
    plt.gcf().set_size_inches((20, 20))
    for i in range(len(imgs)):
        ax = fig.add_subplot(h, h, i + 1)
        if random_order:
            ind = random.randint(0, len(imgs) - 1)
        else:
            ind = i
        img = imgs[ind]
        plt.axis('off')
        plt.imshow(img)
        #
        if not names is None:
            ax.set_title(str(names[ind]))
    if not savepath is None:
        plt.savefig(savepath)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # (batch, h, w, channel)
    sample_mat = np.arange(16).reshape(1, 4, 4, 1)

    # (h, w, channel, output)
    filters = tf.ones(shape=(3, 3, 1, 1))

    # (1, size, size, 1)
    strides = (1, 1, 1, 1)
    fmap = tf.nn.conv2d(sample_mat, filters=filters, strides=strides, padding='VALID')

    # Create Orginal Patail Info (cx, cy, h, w)
    kernel_sizes = [3, 2, 3, 3]
    strides = [1, 2, 2, 1]
    paddings = ['VALID', 'VALID', 'VALID', 'VALID']

    xs, ys = np.meshgrid(np.arange(3), np.arange(2))

    coords = original_rectangle_coords((8, 8), [3, 3], strides=[1, 1], paddings=['valid', 'valid']).reshape((8, 8, -1))
    trgt_coords = coords[0, :, :2]
    print(trgt_coords)

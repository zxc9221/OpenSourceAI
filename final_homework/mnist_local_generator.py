import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
from tqdm import tqdm
from utils import xyxy2xywh, xywh2xyxy, plot_images, images_with_rectangles


def search_mnist_bbox(image):
    """
    Mnist 객체의 bounding box 좌표 정보(x1, y1, x2, y2)를 찾아 반환합니다.
    image: ndarray, shape (28, 28)
    :return:
    (x_min, y_min), (int, int), 이미지 내 mnist 객체의 왼쪽 상단 좌표
    (x_max, y_max), (int, int), 이미지 내 mnist 객체의 오른쪽 하단 좌표
    """
    # sample image
    try:
        axis_0_ind, axis_1_ind = np.where(image > 0)

        y_min = axis_0_ind.min()
        y_max = axis_0_ind.max()
        x_min = axis_1_ind.min()
        x_max = axis_1_ind.max()

    except Exception as e:
        # image에 객체가 없는 경우 에러가 발생 합니다.
        y_min = 0
        y_max = 0
        x_min = 0
        x_max = 0

    return (x_min, y_min), (x_max, y_max)


def random_image_fetcher(xs, canvas_shape=(84, 84)):
    """
    Description:
    임의의 위치와 지정된 크기(canvas shape)에 하나의 mnist 객체가 임의의 위치에 놓여진 데이터를 생성합니다.

    :param xs: ndarray or list, shape=(N, h, w), mnist 정보가 들어있는 image 데이터
    :param canvas_shape: tuple, (int, int), mnist 데이터가 붙여질 이미지의 크기
    :return:
        bg_bucket, ndarray, shape = (N, canvas_shape[0], canvas_shape[1])
        coord_bucket: tuple,
            example)
                ((x1, y1, x2, y2), (x1, y1, x2, y2) ... (x1, y1, x2, y2))
    """

    bg_bucket = []
    coord_bucket = []
    for sample_img in tqdm(xs[:]):
        # Generate h, w background image
        bg = np.random.normal(5, 1, canvas_shape)

        x_w = sample_img.shape[1]
        x_h = sample_img.shape[0]

        # Generate random coords
        rand_x = np.random.randint(0, canvas_shape[1] - x_w)
        rand_y = np.random.randint(0, canvas_shape[0] - x_h)

        # Patch image to background image
        bg[rand_y: rand_y + x_h, rand_x: rand_x + x_w] += sample_img

        # Get mnist object bounding box coordiante
        (obj_x_min, obj_y_min), (obj_x_max, obj_y_max) = search_mnist_bbox(sample_img)

        # add Offset to mnist object coordinate
        obj_x_min += rand_x
        obj_x_max += rand_x
        obj_y_min += rand_y
        obj_y_max += rand_y

        # constraint pixel value from 0 to 255
        bg = np.clip(bg, 0, 255)

        bg_bucket.append(bg)
        coord_bucket.append([obj_x_min, obj_y_min, obj_x_max, obj_y_max])

    return np.array(bg_bucket), np.array(coord_bucket)


def generate_random_coord(canvas_shape, crop_img_size):
    """
    Description:
        배경 이미지에서 crop image 을 붙일수 있는 공간중 임의의 좌표(x,y)를 반환합니다.

    :param canvas_shape: tuple, (height, width), 이미지의 크기
    :param crop_img_size: tuple, (height, width), 이미지의 크기
    :return: coord:, tuple, (x, y), 좌표값
    """
    x_w = crop_img_size[1]
    x_h = crop_img_size[0]

    # Generate random coords
    rand_x = np.random.randint(0, canvas_shape[1] - x_w)
    rand_y = np.random.randint(0, canvas_shape[0] - x_h)
    coord = (rand_x, rand_y)
    return coord


def random_crop(image, sizes):
    """
    image 에서 임의의 위치를 생성해 이미지에서 지정된 크기만큼을 잘라내 붙입니다.

    :param image: ndarray, 2d=(h, w) or 3d=(h, w, ch)
    :param sizes: tuple, (h, w)
    :return: crop_image, ndarray, shape=(size, size)
    """
    canvas_shape = image.shape[0], image.shape[1]
    x, y = generate_random_coord(canvas_shape, sizes)
    crop_img = image[y:y + sizes[0], x:x + sizes[0]]
    return crop_img


def random_crop_images(images, sizes):
    """
    image 에서 임의의 위치를 생성해 이미지에서 지정된 크기만큼을 잘라내 붙입니다.

    :param images: ndarray, 3d=(n, h, w) or 4d=(n, h, w, ch)
    :param sizes_bucket: ndarray, ((crop_h, crop_w), (crop_h, crop_w), ... (crop_h, crop_w))
    :return: crop_images, 3d=(n, h, w) or 4d=(n, h, w, ch)
    """
    cropped_images = []
    for ind, image in enumerate(images):
        cropped_image = random_crop(image, sizes)
        cropped_images.append(cropped_image)
    fetched_images = np.array(cropped_images)
    return fetched_images


def get_random_size(scale_range, ratio_range):
    """
    Description:
    scale 의 범위와 ratio 범위를 받아 random 한 h_size 와 w_size 을 반환합니다.

    :param scale_range: tuple, (scale_min:int, scale_max:int), scale_min > 0  and scale_max > scale_min
    :param ratio_range: tuple, (ratio_min:int, ratio_max:int), ratio_min > 0  and ratio_max > ratio_min
    :return: (h_size, w_size), (h_size:int, w_size:int)
    """
    # generate random scale
    rand_scale = np.random.randint(scale_range[0], scale_range[1])

    # generate random ratio each h, w
    rand_h_ratio = np.random.uniform(ratio_range[0], ratio_range[1])
    rand_w_ratio = np.random.uniform(ratio_range[0], ratio_range[1])

    # calculate h, w size
    h_size = rand_scale * rand_h_ratio
    w_size = rand_scale * rand_w_ratio

    return int(h_size), int(w_size)


def image_resize(image, h_size, w_size):
    """
    Description:
    단일 이미지를 지정된 범위로 resize합니다.

    :param image: ndarray, 2d array, H, W or 3d array, H , W, Ch
    :param h_size: int
    :param w_size: int
    :return:  ndarray, 2d array, H, W or 3d array, H , W, Ch
    """
    dsize = (h_size, w_size)
    return cv2.resize(image, dsize)


def mnist_localization_generator(train_shape, test_shape, background=True, n_sample=None, image_size_range=None,
                                 image_ratio_range=None):
    """
    Description:
    mnist 데이터를 온라인으로 다운받고 다운받은 mnist 데이터를 지정된 크기의 이미지 안에 random 으로 붙입니다.
    그리고 붙여진 위치 정보를 반환합니다.
    train 이미지와 test 이미지 모두 진행됩니다.

    Args:
        train_shape : tuple , (H, W), train image dataset output 형태
        test_shape : tuple , (H, W), test image dataset output 형태
        background : bool, background 정보가 포함되게 할지 아닐지를 결정,
            True 시 이미지와 라벨 마지막에 약 10 정도의 백그라운드 정보를 추가합니다.
        n_samples :  int or None, 몇 개의 데이터만 사용할 지 결정합니다. 디버그시 사용합니다.
        image_size_range : tuple, (size_min:int, size_max:int),
            이미지의 size 범위 값을 제공합니다. 해당 값을 min 값 부터 max 값까지 하나의 값을 추출합니다.
        image_ratio_range : tuple, (ratio_min:float, ratio_maxfloat),
            이미지의 ratio 범위 값을 제공합니다. 해당 값을 min 값 부터 max 값까지 하나의 값을 추출합니다.

    :return:
    train_images: ndarray, 4d array, shape = (N, H, W, C (단 C=1))
    train_cls_true: ndarray, 2d array, shape = (N, 1, 1, 11)
    train_reg_true: ndarray, 2d array, shape =(N, 1, 1 , 4), x1, y1, x2, y2  좌표 형태를 가짐
    test_images: ndarray, 4d array, shape = (N, H, W, C (단 C=1))
    test_cls_true: ndarray, 2d array, shape = (N, 1, 1, 11)
    test_reg_true: ndarray, 2d array, shape = (N, 1, 1, 4), x1, y1, x2, y2  좌표 형태를 가짐

    """
    if background:
        num_classes = 10 + 1
        print('배경 class가 포함되어 class 는 11로 설정 되어 있습니다.')
    else:
        num_classes = 10

    def _xs_processing(xs, canvas_shape):
        """

        Description:
        random 하게 mnist를 지정된 크기의 0 matrix 에 붙여 넣습니다.

        Args:
        :param xs: ndarray, (N, H, W), 입력 이미지 크기, default size는 28 x 28 입니다.
        :param canvas_shape:  mnist 데이터가 붙여질 이미지의 크기
        만약 모든 데이터를 처리시하고 싶을때는 None 을 지정하면 됩니다.

        :return:
            images: ndarray, shape=(N,H,W,C), H, W = canvas shape
            xyxy_coords: ndarray, shape=(H, 4)
        """

        # generate localization training images
        images, xyxy_coord = random_image_fetcher(xs[:], canvas_shape)
        images = np.expand_dims(images, axis=-1)
        return images, xyxy_coord

    def _ys_processing(cls, reg, num_classes):
        """
        Description:
        classification 라벨 데이터의 경우 onehot vector 로 변환, (N) -> (N, 11)
        classification 라벨 데이터의 경우 4차원 형태로 변화 시킴. (N, 11) -> (N, 1, 1, 11)
        regression 라벨 데이터의 경우 x1 y1 x2 y2 을 center x, center y, width, height 로 변환
        regression 라벨 데이터의 경우 4차원 형태로 변화 시킴. (N, 4) -> (N, 1, 1, 4)

        :param cls: ndarray, shape=(N,)
        :param reg: ndarray, shape=(N, 4=(x1, y1, x2, y2))

        :return:
            cls: ndarray, shape=(N, 1, 1, num_classes)
            reg: ndarray, shape=(N, 1, 1, 4=(cx, cy, w, h))
        """
        # generate classification labels
        cls = cls.reshape(-1, 1, 1, 1)[:].astype('float')
        cls = to_categorical(cls, num_classes=num_classes)

        # generate regression labels
        reg = xyxy2xywh(reg)
        reg = reg.reshape(-1, 1, 1, 4)

        return cls, reg

    def _add_background_images(xs, n_bg, noise_data, noise_size):
        """
        Description:
        전체 데이터에 약 10% 비율로 배경 이미지를 생성합니다.
        배경 데이터는 noise 데이터에서 임의로 일정 부분을 잘라낸 후 데이터 임의의 장소에 붙입니다.

        :param xs: ndarray, (N, H, W, 1), 입력 이미지 크기, default size는 84 x 84 입니다.
        :param n_bg: int, 추가할 background 개 수,
        :param noise_data: ndarray,
        :param noise_size: tuple, (int, int)

        :return: ndarray, (N+(N*0.1), H, W, 1)
        """

        canvas_shape = (xs.shape[1], xs.shape[2])

        # background 개 수 만큼 mnist 이미지 중 임의의 위치를 crop 함
        np.random.shuffle(noise_data)
        random_cropped_object = random_crop_images(noise_data[:n_bg], noise_size)
        # crop 된 데이터를 background 이미지에 붙임
        xs_bg, _ = random_image_fetcher(random_cropped_object, canvas_shape)
        xs_bg = xs_bg[..., None]
        return np.concatenate([xs, xs_bg], axis=0)

    def _add_background_labels(cls, reg, n_bg):
        """
        Description:
        기존의 label에 약 10% 비율로 background label 이 추가된 label 을 반환함
        classification 의 경우 background 는 11 로 설정 되어 있음
        regression 의 경우 background 는 0,0,0,0 으로 설정 되어 있음

        :param cls: ndarray, shape=(N, 1, 1, 11)
        :param reg: ndarray, shape=(N, 1, 1, 4)
        :param n_bg: int, 추가할 background 개 수,
        :return:
            cls: ndarray, shape=(N+(N*0.1), 1, 1, 11)
            reg: ndarray, shape=(N+(N*0.1), 1, 1, 4)
        """
        # add train class label and background classes
        bg_cls = np.array([num_classes - 1] * n_bg).reshape((-1, 1, 1, 1))
        bg_cls = to_categorical(bg_cls, num_classes=num_classes)
        cls = np.concatenate([cls, bg_cls], axis=0)

        # add train regression label and background regression
        bg_reg = np.zeros((n_bg, 1, 1, 4))
        reg = np.concatenate([reg, bg_reg], axis=0)
        return cls, reg

    # Image, reg label generator
    (train_xs, train_ys), (test_xs, test_ys) = mnist.load_data()
    train_xs_ori = train_xs.copy()
    test_xs_ori = test_xs.copy()

    # debug mode 시 데이터의 개 수를 줄이기 위해 사용됩니다.
    train_xs, train_ys, test_xs, test_ys = [array[:n_sample] for array in [train_xs, train_ys, test_xs, test_ys]]

    # resize training images with random scale and ratio
    res_train_xs = []
    if not (image_size_range is None) and not (image_ratio_range is None):
        print('train image 을 random 하게 resize 합니다.')
        for train_x in tqdm(train_xs):
            h_size, w_size = get_random_size(image_size_range, image_ratio_range)
            res_train_xs.append(image_resize(train_x, h_size, w_size))
        train_xs = res_train_xs

    # Generate random fetched mnist training dataset
    train_images, train_coords = _xs_processing(train_xs, train_shape)
    train_cls, train_reg = _ys_processing(train_ys, train_coords, num_classes)

    # resize test images with random scale and ratio
    res_test_xs = []
    if not (image_size_range is None) and not (image_ratio_range is None):
        print('train image 을 random 하게 resize 합니다.')
        for test_x in tqdm(test_xs):
            h_size, w_size = get_random_size(image_size_range, image_ratio_range)
            res_test_xs.append(image_resize(test_x, h_size, w_size))
        test_xs = res_test_xs

    # Generate random fetched mnist test dataset
    test_images, test_coords = _xs_processing(test_xs, test_shape)
    test_cls, test_reg = _ys_processing(test_ys, test_coords, num_classes)

    if background:
        print('object 가 없는 background 이미지를 추가합니다.(전체 이미지의 10%)')
        print('background 의 class 는 11로 설정 되어 있습니다.')
        print('background 의 regression 는 (0,0,0,0)로 설정 되어 있습니다.')

        # Object 가 없는 background 이미지를 train, test dataset에 추가합니다.

        # Train
        n_train_bg = int(len(train_images) * 0.1)
        # add train images and background images
        train_images = _add_background_images(train_images, n_train_bg, train_xs_ori, (10, 10))
        # add train labels and background labels
        train_cls, train_reg = _add_background_labels(train_cls, train_reg, n_train_bg)

        # Test
        n_test_bg = int(len(test_images) * 0.1)
        # add train images and background images
        test_images = _add_background_images(test_images, n_test_bg, test_xs_ori, (10, 10))
        # add train labels and background labels
        test_cls, test_reg = _add_background_labels(test_cls, test_reg, n_test_bg)

    return (train_images, train_cls, train_reg), (test_images, test_cls, test_reg)


if __name__ == '__main__':
    # Generate mnist data for localization
    (train_images, train_cls_true, train_reg_true), (test_images, test_cls_true, test_reg_true) = \
        mnist_localization_generator((84, 84), (100, 100), background=True, n_sample=None,
                                     image_size_range=(10, 40),
                                     image_ratio_range=(0.5, 1.5))
    train_reg_true = np.squeeze(xywh2xyxy(np.array([train_reg_true])))
    rected_images = images_with_rectangles(train_images, train_reg_true.reshape(-1, 1, 4))
    plot_images(train_images[-10:])

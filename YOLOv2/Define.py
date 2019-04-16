
TRAIN_DB_XML_DIRS = ['D:/_ImageDataset/VOC2007,2012/VOC2007_Train/Annotations/']
TEST_DB_XML_DIRS = ['D:/_ImageDataset/VOC2007,2012/VOC2007_test/Annotations/']

TENSORBOARD_DIR = './logs/'

#VOC
CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
LABEL_DIC = {k: v for v, k in enumerate(CLASS_NAMES)}

IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416
IMAGE_CHANNEL = 3

GRID_W = 13  #IMAGE_WIDTH / 32
GRID_H = 13  #IMAGE_HEIGHT / 32

GRID_SIZE = int(IMAGE_WIDTH / GRID_W)

BOX_SIZE = 5
CLASSES = len(CLASS_NAMES)

IOU_TH = 0.5

N_ANCHORS = 5
ANCHORS = [[0.57273, 0.677385],
           [1.87446, 2.06253],
           [3.33843, 5.47434],
           [7.88282, 3.52778],
           [9.77052, 9.16828]]

COORD = 5.0
NOOBJ = 0.5

BATCH_SIZE= 16

MAX_ITERS = 100000

LOG_ITER = 100
SAVE_ITER = 1000

DEBUG_SAVE_IMG_COUNT = 5

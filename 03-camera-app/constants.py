# Constants for the training

CONDITIONS = ['like', 'stop', 'peace']

# image size
IMG_SIZE = 64
SIZE = (IMG_SIZE, IMG_SIZE)

# number of color channels we want to use
# set to 1 to convert to grayscale
# set to 3 to use color images
COLOR_CHANNELS = 3

ACTIVATION = 'relu'
LAYER_COUNT = 2
NUM_NEURONS = 64

BATCH_SIZE = 8
EPOCHS = 50

MODEL_PATH = "gesture_recognition.keras"


# Constants for the camera app

CROP_WIDTH = 150
CROP_HEIGHT = 250
CROP_MARGIN = 20
THRESHOLD = 0.65
MIN_HAND_AREA_RATIO = 0.3

VOTE_WINDOW = 5

BACKGROUND_CAPTURE_SECONDS = 2.0

DIFFERENCE_THRESHOLD = 30

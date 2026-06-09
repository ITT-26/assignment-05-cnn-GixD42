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
NUM_NEURONS = 128
DROPOUT = 0.1

BATCH_SIZE = 16
EPOCHS = 50

MODEL_PATH = "gesture_recognition.keras"


# Constants for the camera app

CAMERA_FLIP = True

SELFIE_PATH = "selfies"

SELFIE_TRIGGER_GESTURE = "like"

CROP_WIDTH = 150
CROP_HEIGHT = 200
CROP_MARGIN = 40
THRESHOLD = 0.8
MIN_HAND_AREA_RATIO = 0.4

VOTE_WINDOW = 5

BACKGROUND_CAPTURE_SECONDS = 2.0

DIFFERENCE_THRESHOLD = 20

SELFIE_COUNTDOWN = 3

# seconds to hold a gesture before activating functionality
HOLD_SECOND = 1

# how long to not register a new gesture after the last one
GESTURE_PAUSE_SECONDS = 3
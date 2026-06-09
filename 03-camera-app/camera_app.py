import time
from collections import Counter, deque

import cv2
import numpy as np
from keras.models import load_model

from constants import *


# same preprocessing as in training
def preprocess_image(img):
    if COLOR_CHANNELS == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, SIZE)
    return img_resized


# make an input square where the gestures will be performed
def crop_input_area(frame):
    h, w = frame.shape[:2]

    x1 = CROP_MARGIN
    y1 = h - CROP_HEIGHT - CROP_MARGIN

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))

    x2 = min(w, x1 + CROP_WIDTH)
    y2 = min(h, y1 + CROP_HEIGHT)

    input_area = frame[y1:y2, x1:x2]
    return input_area, (x1, y1, x2, y2)


# captures the background for 2 seconds -> later used for background subtraction
def capture_background(cap):

    # list of captured frames -> average later
    frames = []
    start = time.time()

    # capture frame
    while time.time() - start < BACKGROUND_CAPTURE_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue

        # crop to the input square
        input_area, _ = crop_input_area(frame)
        if input_area is None or input_area.size == 0:
            continue

        # add to background frames
        frames.append(input_area.astype(np.float32))

        # remaining time
        remaining = max(0.0, BACKGROUND_CAPTURE_SECONDS -
                        (time.time() - start))

        # put text to show that the background is being captured
        cv2.putText(
            frame,
            f"Calibrating background, try not to move... {remaining:.1f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        # show the frame
        cv2.imshow("Gesture Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # return average background frame
    return np.mean(frames, axis=0).astype(np.uint8)


# should detect hand by doing background subtraction -> crop around hand
def crop_hand_with_static_bg(input_area, background):
    # if somethings missing --> return None
    if input_area is None or background is None:
        return None, None

    # background subtraction -> to grayscale
    diff = cv2.absdiff(input_area, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # simple mask
    _, mask = cv2.threshold(gray, DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY)

    # contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # largest contour -> assume it's the hand
    c = max(contours, key=cv2.contourArea)

    # is the contour big enough to be the hand: no -> return None
    min_area = input_area.shape[0] * input_area.shape[1] * MIN_HAND_AREA_RATIO
    if cv2.contourArea(c) < min_area:
        return None, None

    # crop box around the hand
    x, y, w, h = cv2.boundingRect(c)
    hand = input_area[y:y+h, x:x+w]
    if hand.size == 0:
        return None, None

    return hand, (x, y, w, h)


def main():
    # load the model and labels
    model = load_model(MODEL_PATH)
    labels = CONDITIONS

    # get video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera couldn't be opened.")

    # background capture
    background = capture_background(cap)

    # for voting
    last_label = None
    label_history = deque(maxlen=VOTE_WINDOW)

    # loop
    while True:

        # get frame
        ret, frame = cap.read()
        if not ret:
            break

        # crop to input square and crop hand with background subtraction
        input_area, (rx1, ry1, rx2, ry2) = crop_input_area(frame)
        hand_crop, hand_box = crop_hand_with_static_bg(
            input_area, background)

        # standard label and confidence for when no hand is detected
        raw_label = "unknown"
        confidence = 0.0

        # Only recognize if a hand box was found
        if hand_box is not None and hand_crop is not None:

            # preprocess the hand crop and predict with the model
            input_frame = preprocess_image(hand_crop)
            input_frame = np.expand_dims(input_frame, axis=0)

            # get prediction and confidence for the most likely class
            prediction = model.predict(input_frame, verbose=0)[0]
            idx = int(np.argmax(prediction))
            confidence = float(prediction[idx])

            # if confidence is above the threshold, use the predicted label, otherwise keep it as "unknown"
            if confidence >= THRESHOLD:
                raw_label = labels[idx]

        # in history for voting
        label_history.append(raw_label)

        # voted label is the most common label in the history
        voted_label = Counter(label_history).most_common(1)[0][0]

        # draw the input square, hand box and label on the original frame
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

        if hand_box is not None:
            hx, hy, hw, hh = hand_box
            cv2.rectangle(
                frame,
                (rx1 + hx, ry1 + hy),
                (rx1 + hx + hw, ry1 + hy + hh),
                (255, 0, 0),
                2,
            )

        # output text
        cv2.putText(
            frame,
            f"Gesture: {voted_label} ({confidence:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # show the frame
        cv2.imshow("Gesture Camera", frame)

        # read input
        key = cv2.waitKey(1) & 0xFF

        # q to quit
        if key == ord("q"):
            break

        # r to recalibrate background
        if key == ord("r"):
            background = capture_background(cap)
            label_history.clear()

    # close everything
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

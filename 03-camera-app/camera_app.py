import time
from datetime import datetime
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from keras.models import load_model

from constants import *


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH_REL = BASE_DIR / MODEL_PATH
SELFIE_PATH_REL = BASE_DIR / SELFIE_PATH


# same preprocessing as in training
def preprocess_image(img):
    if COLOR_CHANNELS == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, SIZE)
    return img_resized


# make an input square where the gestures will be performed
def crop_input_area(frame):
    h, w = frame.shape[:2]

    x1 = w - CROP_WIDTH - CROP_MARGIN
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

        if CAMERA_FLIP:
            frame = cv2.flip(frame, 1)

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
            f"Calibrating background... {remaining:.1f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
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


# sepia filter from https://amin-ahmadi.com/2016/03/24/sepia-filter-opencv/
def sepia_filter(frame):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia


def main():
    # load the model and labels
    model = load_model(str(MODEL_PATH_REL))
    labels = CONDITIONS

    # get video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera couldn't be opened.")

    # background capture
    background = capture_background(cap)

    # for voting
    label_history = deque(maxlen=VOTE_WINDOW)

    # for activating funcitonality
    hold_label = None
    hold_start_time = 0
    last_gesture_time = 0

    # for selfie countdown
    selfie_trigger = False
    selfie_countdown_end = 0

    # sepia flag
    sepia = False

    # loop
    while True:

        # get frame
        ret, frame = cap.read()
        if not ret:
            break

        if CAMERA_FLIP:
            frame = cv2.flip(frame, 1)

        # apply sepia
        if sepia:
            frame = sepia_filter(frame)

        # frame without ui elements
        clean_frame = frame.copy()

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

        # logic for activating functionality
        now = time.time()

        # if a selfie is about to be taken, nothing else should be happening
        if selfie_trigger:

            # remaining time
            remaining = selfie_countdown_end - now
            remaining_sec = int(np.ceil(remaining))

            # show countdown
            if remaining_sec > 0:
                cv2.putText(
                    clean_frame,
                    f"Taking selfie in {remaining_sec}...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            # countdown is over -> take selfie
            if remaining <= 0:
                # timestamp for unique filename
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = SELFIE_PATH_REL / f"selfie_{ts}.jpg"
                # save clean frame
                cv2.imwrite(str(out_path), clean_frame)
                selfie_trigger = False

            # don't show anything else while countdown is active
            cv2.imshow("Gesture Camera", clean_frame)

            # input handling as normal
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                background = capture_background(cap)
                label_history.clear()

            # skip the rest of the loop -> no other functionality during countdown
            continue

        # logic for activating functionality -> gesture needs to be held for a certain amount of time
        if voted_label == "unknown":
            hold_label = None
            hold_start_time = 0

        # first time the label is seen -> timer starts
        elif voted_label != hold_label:
            hold_label = voted_label
            hold_start_time = now

        # timer is running -> check for how long
        else:
            # how long the same gesture
            hold_duration = now - hold_start_time
            cooldown_check = (now - last_gesture_time) >= GESTURE_PAUSE_SECONDS

            # gesture held long enough
            if hold_duration >= HOLD_SECOND and cooldown_check:

                # debug
                # print(f"GESTURE ACTIVATED: {hold_label}")

                # gesture specific functionality
                if hold_label == SELFIE_TRIGGER_GESTURE:
                    selfie_trigger = True
                    selfie_countdown_end = now + SELFIE_COUNTDOWN

                elif hold_label == SEPIA_TRIGGER_GESTURE:
                    sepia = not sepia

                # cooldown for gesture activation
                last_gesture_time = now
                hold_label = None
                hold_start_time = 0

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

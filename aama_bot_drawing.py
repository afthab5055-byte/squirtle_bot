import cv2
import mediapipe as mp
import numpy as np
import math

# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width == 0 or height == 0:
    width = 1280
    height = 720

# ============================================================
# CANVAS
# ============================================================

canvas = np.zeros(
    (height, width, 3),
    dtype=np.uint8
)

# ============================================================
# COLORS - BGR
# ============================================================

COLORS = {
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255)
}

current_color = COLORS["blue"]

# ============================================================
# DRAWING SETTINGS
# ============================================================

thickness = 5

tool = "freehand"

# Previous freehand point
prev_x = None
prev_y = None

# Shape starting point
start_x = None
start_y = None

# Whether currently drawing
drawing = False

# ============================================================
# FUNCTIONS
# ============================================================

def draw_triangle(img, x1, y1, x2, y2, color, thickness):

    left = min(x1, x2)
    right = max(x1, x2)

    top = min(y1, y2)
    bottom = max(y1, y2)

    center_x = (left + right) // 2

    points = np.array([
        [center_x, top],
        [left, bottom],
        [right, bottom]
    ], dtype=np.int32)

    cv2.polylines(
        img,
        [points],
        True,
        color,
        thickness
    )


def draw_shape(img, x1, y1, x2, y2):

    if tool == "line":

        cv2.line(
            img,
            (x1, y1),
            (x2, y2),
            current_color,
            thickness
        )

    elif tool == "rectangle":

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            current_color,
            thickness
        )

    elif tool == "circle":

        radius = int(
            math.sqrt(
                (x2 - x1) ** 2 +
                (y2 - y1) ** 2
            )
        )

        cv2.circle(
            img,
            (x1, y1),
            radius,
            current_color,
            thickness
        )

    elif tool == "triangle":

        draw_triangle(
            img,
            x1,
            y1,
            x2,
            y2,
            current_color,
            thickness
        )


def draw_toolbar(img):

    # Toolbar background
    cv2.rectangle(
        img,
        (0, 0),
        (width, 100),
        (35, 35, 35),
        -1
    )

    cv2.putText(
        img,
        "AIR DRAWING",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        "Tool: " + tool.upper(),
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        current_color,
        2
    )

    # Colors
    colors = [
        ("1", COLORS["blue"]),
        ("2", COLORS["red"]),
        ("3", COLORS["green"]),
        ("4", COLORS["yellow"]),
        ("5", COLORS["white"])
    ]

    x = 300

    for key, color in colors:

        cv2.rectangle(
            img,
            (x, 15),
            (x + 45, 60),
            color,
            -1
        )

        cv2.putText(
            img,
            key,
            (x + 17, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        x += 60


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Camera error")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    # ========================================================
    # HAND DETECTED
    # ========================================================

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # ----------------------------------------------------
        # INDEX FINGER
        # ----------------------------------------------------

        index = hand.landmark[8]

        x = int(index.x * width)
        y = int(index.y * height)

        # ----------------------------------------------------
        # THUMB
        # ----------------------------------------------------

        thumb = hand.landmark[4]

        tx = int(thumb.x * width)
        ty = int(thumb.y * height)

        # ----------------------------------------------------
        # PINCH DETECTION
        # ----------------------------------------------------

        distance = math.sqrt(
            (x - tx) ** 2 +
            (y - ty) ** 2
        )

        # Adaptive threshold
        pinch_threshold = 45

        pinch = distance < pinch_threshold

        # ====================================================
        # DRAW FINGERTIP
        # ====================================================

        if pinch:

            cv2.circle(
                frame,
                (x, y),
                12,
                current_color,
                -1
            )

        else:

            cv2.circle(
                frame,
                (x, y),
                10,
                (255, 255, 255),
                2
            )

        # ====================================================
        # START DRAWING
        # ====================================================

        if pinch and not drawing:

            drawing = True

            start_x = x
            start_y = y

            prev_x = x
            prev_y = y

        # ====================================================
        # CURRENTLY DRAWING
        # ====================================================

        if pinch and drawing:

            # ------------------------------------------------
            # FREEHAND
            # ------------------------------------------------

            if tool == "freehand":

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (x, y),
                    current_color,
                    thickness
                )

                prev_x = x
                prev_y = y

            # ------------------------------------------------
            # ERASER
            # ------------------------------------------------

            elif tool == "eraser":

                cv2.circle(
                    canvas,
                    (x, y),
                    35,
                    (0, 0, 0),
                    -1
                )

            # ------------------------------------------------
            # SHAPES
            # ------------------------------------------------

            elif tool in [
                "line",
                "rectangle",
                "circle",
                "triangle"
            ]:

                # Make temporary preview
                preview = canvas.copy()

                draw_shape(
                    preview,
                    start_x,
                    start_y,
                    x,
                    y
                )

                # Put preview on camera
                frame = cv2.addWeighted(
                    frame,
                    0.5,
                    preview,
                    0.5,
                    0
                )

        # ====================================================
        # RELEASE PINCH
        # ====================================================

        if not pinch and drawing:

            # Finish shape
            if tool in [
                "line",
                "rectangle",
                "circle",
                "triangle"
            ]:

                draw_shape(
                    canvas,
                    start_x,
                    start_y,
                    x,
                    y
                )

            drawing = False

            start_x = None
            start_y = None

            prev_x = None
            prev_y = None

    else:

        drawing = False

        start_x = None
        start_y = None

        prev_x = None
        prev_y = None

    # ========================================================
    # COMBINE CAMERA + CANVAS
    # ========================================================

    gray = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    mask_inv = cv2.bitwise_not(mask)

    camera_background = cv2.bitwise_and(
        frame,
        frame,
        mask=mask_inv
    )

    drawing_layer = cv2.bitwise_and(
        canvas,
        canvas,
        mask=mask
    )

    output = cv2.add(
        camera_background,
        drawing_layer
    )

    # ========================================================
    # TOOLBAR
    # ========================================================

    draw_toolbar(output)

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    cv2.putText(
        output,
        "PINCH INDEX + THUMB TO DRAW",
        (20, height - 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        "F Freehand | L Line | R Rectangle | O Circle | T Triangle | E Eraser",
        (20, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        "1 Blue | 2 Red | 3 Green | 4 Yellow | 5 White | C Clear | S Save | Q Quit",
        (20, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Virtual Air Drawing",
        output
    )

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # Colors
    if key == ord("1"):
        current_color = COLORS["blue"]

    elif key == ord("2"):
        current_color = COLORS["red"]

    elif key == ord("3"):
        current_color = COLORS["green"]

    elif key == ord("4"):
        current_color = COLORS["yellow"]

    elif key == ord("5"):
        current_color = COLORS["white"]

    # Tools
    elif key == ord("f"):
        tool = "freehand"

    elif key == ord("l"):
        tool = "line"

    elif key == ord("r"):
        tool = "rectangle"

    elif key == ord("o"):
        tool = "circle"

    elif key == ord("t"):
        tool = "triangle"

    elif key == ord("e"):
        tool = "eraser"

    # Clear
    elif key == ord("c"):

        canvas = np.zeros(
            (height, width, 3),
            dtype=np.uint8
        )

    # Save
    elif key == ord("s"):

        cv2.imwrite(
            "air_drawing.png",
            canvas
        )

        print("Drawing saved as air_drawing.png")

    # Quit
    elif key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
hands.close()

import cv2
import numpy as np
import math
import glob


# ============================================================
# CAMERA SETTINGS
# ============================================================

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Try camera devices in this order
CAMERA_DEVICES = [
    "/dev/video0",
    "/dev/video1",
    "/dev/video2",
    "/dev/video3",
    "/dev/video4",
    "/dev/video5"
]


# ============================================================
# ARUCO SETTINGS
# ============================================================

ARUCO_DICT = cv2.aruco.DICT_4X4_50

ROBOT_MARKER_ID = 0

WAYPOINT_THRESHOLD = 35


# ============================================================
# ARUCO SETUP
# ============================================================

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    ARUCO_DICT
)

aruco_parameters = cv2.aruco.DetectorParameters()

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    aruco_parameters
)


# ============================================================
# VARIABLES
# ============================================================

waypoints = []

current_waypoint = 0

navigation_started = False


# ============================================================
# CAMERA FINDER
# ============================================================

def find_camera():

    print("Searching for available camera...")

    devices = glob.glob("/dev/video*")

    if not devices:

        print("No video devices found.")

        return None


    # Sort devices numerically
    def device_number(device):

        try:
            return int(
                device.replace("/dev/video", "")
            )

        except:

            return 999


    devices.sort(
        key=device_number
    )


    for device in devices:

        print(
            f"Trying camera: {device}"
        )

        cap = cv2.VideoCapture(
            device,
            cv2.CAP_V4L2
        )

        if not cap.isOpened():

            cap.release()

            continue


        # Try MJPEG
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        cap.set(
            cv2.CAP_PROP_FPS,
            FPS
        )

        # Reduce buffering
        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )


        # Try to actually read a frame
        ret, frame = cap.read()


        if ret and frame is not None:

            print(
                f"Camera selected: {device}"
            )

            print(
                f"Resolution: "
                f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
                f"x"
                f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            )

            print(
                f"FPS: "
                f"{cap.get(cv2.CAP_PROP_FPS)}"
            )

            return cap


        cap.release()


    print("Could not find a working camera.")

    return None


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global waypoints
    global navigation_started
    global current_waypoint


    if event == cv2.EVENT_LBUTTONDOWN:

        waypoints.append(
            (x, y)
        )

        navigation_started = False

        current_waypoint = 0

        print(
            f"Waypoint added: ({x}, {y})"
        )


# ============================================================
# OPEN CAMERA
# ============================================================

cap = find_camera()


if cap is None:

    print()
    print(
        "ERROR: No working camera found."
    )
    print(
        "Connect a camera and try again."
    )

    exit()


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = "Aama Bot Interface"


cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    FRAME_WIDTH,
    FRAME_HEIGHT
)


cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:


    # --------------------------------------------------------
    # Read camera
    # --------------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        print(
            "Camera frame lost."
        )

        break


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    frame = cv2.resize(
        frame,
        (
            FRAME_WIDTH,
            FRAME_HEIGHT
        )
    )


    # ========================================================
    # ARUCO DETECTION
    # ========================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    corners, ids, rejected = (
        aruco_detector.detectMarkers(
            gray
        )
    )


    robot_found = False

    robot_x = 0

    robot_y = 0

    robot_angle = 0


    # ========================================================
    # MARKER PROCESSING
    # ========================================================

    if ids is not None:


        cv2.aruco.drawDetectedMarkers(
            frame,
            corners,
            ids
        )


        for i, marker_id in enumerate(
            ids.flatten()
        ):


            if marker_id == ROBOT_MARKER_ID:


                robot_found = True


                marker_corners = (
                    corners[i][0]
                )


                # --------------------------------------------
                # CENTER
                # --------------------------------------------

                robot_x = int(
                    np.mean(
                        marker_corners[:, 0]
                    )
                )


                robot_y = int(
                    np.mean(
                        marker_corners[:, 1]
                    )
                )


                # --------------------------------------------
                # HEADING
                # --------------------------------------------

                top_left = (
                    marker_corners[0]
                )

                top_right = (
                    marker_corners[1]
                )


                dx = (
                    top_right[0]
                    - top_left[0]
                )


                dy = (
                    top_right[1]
                    - top_left[1]
                )


                robot_angle = math.degrees(
                    math.atan2(
                        dy,
                        dx
                    )
                )


                # --------------------------------------------
                # CENTER
                # --------------------------------------------

                cv2.circle(
                    frame,
                    (
                        robot_x,
                        robot_y
                    ),
                    7,
                    (0, 255, 0),
                    -1
                )


                # --------------------------------------------
                # HEADING
                # --------------------------------------------

                heading_length = 70


                heading_x = int(
                    robot_x
                    + heading_length
                    * math.cos(
                        math.radians(
                            robot_angle
                        )
                    )
                )


                heading_y = int(
                    robot_y
                    + heading_length
                    * math.sin(
                        math.radians(
                            robot_angle
                        )
                    )
                )


                cv2.arrowedLine(
                    frame,
                    (
                        robot_x,
                        robot_y
                    ),
                    (
                        heading_x,
                        heading_y
                    ),
                    (0, 255, 0),
                    3,
                    tipLength=0.3
                )


                # --------------------------------------------
                # INFORMATION
                # --------------------------------------------

                cv2.putText(
                    frame,
                    f"Robot ID: {ROBOT_MARKER_ID}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"Position: "
                    f"({robot_x}, {robot_y})",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"Angle: "
                    f"{robot_angle:.1f}",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


    # ========================================================
    # DRAW WAYPOINTS
    # ========================================================

    for i, point in enumerate(
        waypoints
    ):


        x, y = point


        if (
            navigation_started
            and i == current_waypoint
        ):


            cv2.circle(
                frame,
                (x, y),
                13,
                (0, 0, 255),
                -1
            )


        else:


            cv2.circle(
                frame,
                (x, y),
                8,
                (0, 255, 255),
                -1
            )


        cv2.putText(
            frame,
            str(i + 1),
            (
                x + 12,
                y - 12
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # ========================================================
    # DRAW PATH
    # ========================================================

    if len(waypoints) > 1:


        for i in range(
            len(waypoints) - 1
        ):


            cv2.line(
                frame,
                waypoints[i],
                waypoints[i + 1],
                (255, 150, 0),
                3
            )


    # ========================================================
    # NAVIGATION
    # ========================================================

    if (
        navigation_started
        and robot_found
        and current_waypoint
        < len(waypoints)
    ):


        target_x, target_y = (
            waypoints[
                current_waypoint
            ]
        )


        dx = (
            target_x - robot_x
        )


        dy = (
            target_y - robot_y
        )


        distance = math.sqrt(
            dx * dx +
            dy * dy
        )


        target_angle = math.degrees(
            math.atan2(
                dy,
                dx
            )
        )


        # Draw target line

        cv2.line(
            frame,
            (
                robot_x,
                robot_y
            ),
            (
                target_x,
                target_y
            ),
            (0, 0, 255),
            2
        )


        cv2.putText(
            frame,
            f"Target: "
            f"{current_waypoint + 1}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


        cv2.putText(
            frame,
            f"Distance: "
            f"{distance:.1f}px",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


        cv2.putText(
            frame,
            f"Target Angle: "
            f"{target_angle:.1f}",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


        # Waypoint reached

        if distance < WAYPOINT_THRESHOLD:


            print(
                f"Waypoint "
                f"{current_waypoint + 1}"
                f" reached!"
            )


            current_waypoint += 1


            if (
                current_waypoint
                >= len(waypoints)
            ):


                print(
                    "All waypoints completed!"
                )


                navigation_started = False


    # ========================================================
    # STATUS
    # ========================================================

    if robot_found:


        cv2.putText(
            frame,
            "ROBOT DETECTED",
            (
                20,
                FRAME_HEIGHT - 55
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    else:


        cv2.putText(
            frame,
            "ROBOT MARKER NOT DETECTED",
            (
                20,
                FRAME_HEIGHT - 55
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    # ========================================================
    # CONTROLS
    # ========================================================

    cv2.putText(
        frame,
        "Click: Waypoint | "
        "S: Start | "
        "C: Clear | "
        "R: Reset | "
        "Q: Quit",
        (
            20,
            FRAME_HEIGHT - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    elif key == ord("s"):


        if len(waypoints) > 0:


            navigation_started = True

            current_waypoint = 0


            print(
                "Navigation started."
            )


        else:


            print(
                "Add at least one waypoint."
            )


    elif key == ord("c"):


        waypoints.clear()

        navigation_started = False

        current_waypoint = 0


        print(
            "Waypoints cleared."
        )


    elif key == ord("r"):


        navigation_started = False

        current_waypoint = 0


        print(
            "Navigation reset."
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Camera closed.")
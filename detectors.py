import cv2
import numpy as np

def detect_trafficlight_color(video_frame: np.ndarray, scale_factor: float) -> tuple[str, tuple]:
    """
        this function check for every section of the traffic light (top, center, bottom) if it is active, then return the color of the active section

        :param video_frame:
        :return: str
    """

    # HARD-CODED POSITION, CHANGE THEM IF NEEDED
    rect = (int(1810*scale_factor), int(160*scale_factor), int(110*scale_factor), int(250*scale_factor))
    x, y, w, h = rect

    # crop the traffic light box
    traffic_light = video_frame[y - int(h / 2): y + int(h / 2), x - int(w / 2):x + int(w / 2)].copy()
    h, w, _ = traffic_light.shape

    # Crop the 3 traffic light section
    red = traffic_light[:int(h / 3), :]
    yellow = traffic_light[int(h / 3):h - int(h / 3), :]
    green = traffic_light[h - int(h / 3):, :]

    colors = {
        "yellow": yellow,
        "green": green,
        "red": red, #need to be the last color to be processed
    }

    tl_color = ""
    min_white_pixels = 500
    for name, arr in colors.items():
        # apply tresholding to detect the active color (the red one do not pass the thresholding)
        gris = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        pimg = cv2.medianBlur(gris, 7)
        _, thresholded_image = cv2.threshold(pimg, 110, 255, cv2.THRESH_BINARY)

        # count the white pixel (if there are some, it means that that color is active)
        white_pixels = np.sum(thresholded_image == 255)

        if white_pixels > min_white_pixels:
            tl_color = name

        # the red one is the only one that do not pass the thresholding, (no white pixels when active)
        # is active whene either green and yellow are not active
        if name == "red" and tl_color == "":
            tl_color = name

    colors_bgr = {
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "red": (0, 0, 255)
    }

    return tl_color, colors_bgr[tl_color]


def detect_line(video_frame: np.ndarray, past_line: list, scale_factor: float) -> list:
    """

    this function mask the frame to a specific section where the line could be,
    then the frame is converted to gray, Canny filter is applied and the Hough Line Transform at the end.
    some line have been detected, or none, to choose the line do some checks:

        -only the almost horizontal line
        -only the one that are far from the borders (since i have a mask the Hough Line Transform detect
            the bord of a mask as a line, i don't consider those lines)

    if both past and new line are valid i kee



    :param video_frame:
    :param past_line:
    :return: list
    """

    #====MASKING=====
    height, width = video_frame.shape[:2]
    slope1, intercept1 = 0.03, int(920*scale_factor)  # upper line
    slope2, intercept2 = 0.03, int(770*scale_factor)  # bottom line
    slope3, intercept3 = -0.3, int(1000*scale_factor)  # inclined line, to divide the lanes
    def line1(x):
        return slope1 * x + intercept1

    def line2(x):
        return slope2 * x + intercept2

    def line3(x):
        return slope3 * x + intercept3

    mask1 = video_frame.copy()
    for x in range(width):
        y_line = line1(x)
        mask1[int(y_line):, x] = 0

    mask2 = mask1.copy()
    # Set pixels above the second line to black in mask2
    for x in range(width):
        y_line = line2(x)
        mask2[:int(y_line), x] = 0

    mask3 = mask2.copy()
    # Set pixels to the left of the third line to black in mask3 (final mask)
    for y in range(height):
        x_line = line3(y)
        mask3[y, :int(x_line)] = 0

    _, final_mask = cv2.threshold(mask3, 0, 255, cv2.THRESH_BINARY)

    masked_image = cv2.bitwise_and(video_frame, video_frame, mask=final_mask[:, :, 0])
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Mostra bordi

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=400, maxLineGap=50)

    angle_threshold = 10

    max_line = [0, 0, 0, 0]  # the longest line that pass the contitions
    horizontal = []
    final_line = past_line.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Avoid division to 0
            if x2 - x1 == 0:
                angle = 90
            else:
                # angle in degree
                angle = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))
                angle = abs(angle)

            # Save only the horizontal lines
            if abs(angle) < angle_threshold:
                horizontal.append(line)

        # Ignore the lines of the border of the mask (the nearest and the farest one)
        horizontal = sorted(horizontal, key=lambda l: min(l[0][1], l[0][3]))

        # top border line
        top_line = horizontal[0][0]
        tx1, ty1, tx2, ty2 = top_line

        # bottom border line
        bottom_line = horizontal[-1][0]
        bx1, by1, bx2, by2 = bottom_line

        # all the lines with a distance > 40 to the border line are accepted
        epsilon = int(40*scale_factor)
        max_length = 0

        for line in horizontal:
            x1, y1, x2, y2 = line[0]

            # consider only the lines far enough from the borders
            if (y1 > ty1 + epsilon and y2 > ty2 + epsilon) and (y1 < by1 - epsilon and y2 < by2 - epsilon):
                line_length = np.abs(x2 - x1)
                if line_length > max_length:  # save only the longest line
                    max_length = line_length
                    max_line = line[0]

        if not np.array_equal(past_line, np.array([0, 0, 0, 0])):  # if the past line was valid
            if not np.array_equal(max_line, np.array([0, 0, 0, 0])):  # if the max_line is valid
                # compute the mean between the past line and the new line
                final_line = np.concatenate([[past_line], [max_line]], axis=0)
                final_line = np.mean(final_line, axis=0)
                final_line = list(map(int, final_line))
            else:
                # if no line detected, keep the past line
                final_line = past_line
        elif not np.array_equal(max_line, np.array([0, 0, 0, 0])):
            # if the past line wasnt valid, keep only the new line, if valid
            final_line = max_line


    #extend the line in lenght
    x1, y1, x2, y2 = final_line
    #if the lenght is already the one i want is not neccessary
    if not np.array_equal(final_line, np.array([0, 0, 0, 0])) and x1 != int(700*scale_factor) and x2 != width:

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        starting_x = int(700*scale_factor)
        ending_x = width
        starting_y = slope * starting_x + intercept
        ending_y = slope * ending_x + intercept
        final_line = [int(starting_x), int(starting_y), int(ending_x), int(ending_y)]

    return final_line

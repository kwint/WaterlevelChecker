import cv2
import numpy as np


def nothing(x):
    pass


def init():
    #Create sliders
    cv2.namedWindow("slider", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("low", "slider", 0, 255, nothing)
    cv2.createTrackbar("high", "slider", 0, 255, nothing)
    cv2.setTrackbarPos("low", "slider", 146)
    cv2.setTrackbarPos("high", "slider", 4)


# Return edges
def get_edges(img, low, high):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, low, high)
    return canny

# Returns Threshold image
def get_threshold(img, threshold, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


# Finds ground/floor on the image
def get_edge_hight(img):
    img_edge = img[0:756, 0:10]

    edge_canny = get_edges(img_edge, 57, 120)
    im, contours_edge, hierarchy_edge = cv2.findContours(edge_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print("Ground level found on: ", contours_edge[0][0][0][1])
    return contours_edge[0][0][0][1]


# Returns water level
def get_water_level(img, row):

    img_thresh = get_threshold(img, 97, 4)

    for pixel in range(0, img_thresh.shape[1]):
        if img_thresh[pixel][row] == 0:
            print("water level found!", pixel)
            img[pixel][row] = [0, 255, 0]
            return pixel
        else:
            img[pixel][row] = [255, 0, 0]

# Returns position, and width of top
def get_top(img):
    # Build mask array
    lower_unit = np.array([105, 57, 32])
    upper_unit = np.array([137, 255, 110])

    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Filter colors
    mask = cv2.inRange(hsv, lower_unit, upper_unit)
    res = cv2.bitwise_and(img, img, mask=mask)

    # Get threshold
    thresh = get_threshold(res, 20, 5)

    # Recognize contoours
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in range(0, len(contours)):
        if cv2.contourArea(contours[cnt]) > 200:
            x, y, w, h = cv2.boundingRect(contours[cnt])

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            print("Found top on: ", x, y, "Width:", w, "Height: ", h)
            return y + h, x, w,

# Calculate cm to pixel ratio
def calc_cmpx(pixels, cm):
    return cm / pixels


init()
diameter = []
water_found = False
while True:

    # Read Image
    img = cv2.imread("images/fles1.jpg")

    # Get the height of the ground
    edge_height = get_edge_hight(img)

    # Find the top of the bottle, returns the hight and the width (4 cm in real-life)
    top_y, top_x, top_width = get_top(img[0:int(img.shape[1] / 2)])
    cmpx = calc_cmpx(top_width, 4.2)
    print(cmpx)

    # Find the water level in the image between the ground (edge) and the top of the bottle. If function returns error
    # there is no water level and program runs again
    try:
        water_level = get_water_level(img[top_y:edge_height], int(top_x + (top_width / 2))) + top_y
        water_found = True
    except Exception:
        print("Couldn't find water level, bottle empty?")
        water_found = False

    # When water level is found run the following code
    if water_found:

        # Crop the image between the water level and the ground and make it binary
        img_crop = img[top_y:water_level, 0:567]
        img_crop_thresh = get_threshold(img_crop, cv2.getTrackbarPos("low", "slider"),
                                        cv2.getTrackbarPos("high", "slider"))
        cv2.imshow("thress", img_crop_thresh)

        # Draw the water level and the edge height to the image
        cv2.line(img, (0, water_level + 1), (img.shape[1], water_level + 1), (0, 0, 255), 2)
        cv2.line(img, (0, top_y), (img.shape[1], top_y), (0, 255, 0), 2)

        # Get the diameter of the bottle for each pixel row
        for pixel_row in range(0, img_crop_thresh.shape[0]):
            # checks for the first black pixel from the left
            for pixel_left in range(0, int(img_crop_thresh.shape[1] / 2), 1):
                if img_crop_thresh[pixel_row][pixel_left] == 0:
                    break
            # check for the first black pixel form the right
            for pixel_right in range(int(img_crop_thresh.shape[1] - 80), int(img_crop_thresh.shape[1] / 2), -1):
                if img_crop_thresh[pixel_row][pixel_right] == 0:
                    break
            # Compensate for cutting the image in half
            pixel_right -= int(img_crop_thresh.shape[1] / 2)

            # Calculate the diameter
            diameter.append((pixel_right + (567 / 2 - pixel_left)) * cmpx)

            # print edge to image then print it to the screen
            img[pixel_row + top_y][pixel_right + int(img_crop_thresh.shape[1] / 2)] = [0, 0, 0]
            img[pixel_row + top_y][pixel_left] = [0, 0, 0]
            cv2.imshow("img", img)
            cv2.waitKey(5)

        # calculate the volume per pixel row
        volume_slice = []
        for i in range(0, len(diameter)):
            volume_slice.append((((diameter[i] / 2) ** 2) * np.pi) * cmpx)

        # Calculate the total volume and print it to the console
        volume = round(sum(volume_slice))
        print("Volume between lines: ", volume, " ml")
        print("Volume of bottle ", 500 - volume)

        # Program waits until esc is pressed, then exits
        if cv2.waitKey(0) in {1048603, 27}:
            break

    # Quit program when esc is pressed, for windows and linux
    if cv2.waitKey(1) in {1048603, 27}:
        cv2.destroyAllWindows()
        break
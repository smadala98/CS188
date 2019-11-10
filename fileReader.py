# Program To Read video
# and Extract Frames
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template

# Function to extract frames
def FrameCapture(path):
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Saves the frames with frame-count
        cv2.imwrite("frame%d.jpg" % count, image)

        count += 1


def plot(image_path):
    image = cv2.imread(image_path, 0)
    template = cv2.imread("frame8template.jpg", 0)
    result = match_template(image, template, pad_input=True)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y =ij[::-1]
    # plt.imshow(result)
    # plt.plot(x, y)
    # plt.show()
    result = np.where(result == np.amax(result))
    listOfCordinates = (result[0][0], result[1][0])
    translation = np.full((2, 3), 0, dtype=float)
    translation[0][0] = 1
    translation[0][2] = listOfCordinates[0] * -1
    translation[1][1] = 1
    translation[1][2] = listOfCordinates[1] * -1
    return cv2.warpAffine(image, translation, image.shape)

# Driver Code
if __name__ == '__main__':
    # Calling the function
    # FrameCapture("IMG_0412.MOV")
    # x = []
    # y = []
    # for i in range(244):
    #     coords = plot("frame%d.jpg" % i)
    #     y.append(coords[0])
    #     x.append(coords[1])
    # print(x)
    # print(y)
    # plt.plot(x, y)
    # plt.show()
    #plot("frame0.jpg")
    final_image = plot("frame0.jpg")
    for i in range(1, 244):
        shifted_image = plot("frame%d.jpg" % i)
        final_image = np.add(final_image, shifted_image)
    cv2.imshow("pic", final_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
import numpy as np
import cv2
import PySimpleGUI as sg
import os.path

class ImageColorizer:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        # Initialize the CLAHE object with the given parameters
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        self.version = '7 June 2020'

        self.prototxt = r'model/colorization_deploy_v2.prototxt'
        self.model = r'model/colorization_release_v2.caffemodel'
        self.points = r'model/pts_in_hull.npy'
        self.points = os.path.join(os.path.dirname(__file__), self.points)
        self.prototxt = os.path.join(os.path.dirname(__file__), self.prototxt)
        self.model = os.path.join(os.path.dirname(__file__), self.model)
        if not os.path.isfile(self.model):
            sg.popup_scrolled('Missing model file', 'You are missing the file "colorization_release_v2.caffemodel"',
                            'Download it and place into your "model" folder', 'You can download this file from this location:\n', r'https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1')
            exit()
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)     # load model from disk
        self.pts = np.load(self.points)

        # add the cluster centers as 1x1 convolutions to the model
        self.class8 = self.net.getLayerId("class8_ab")
        self.conv8 = self.net.getLayerId("conv8_313_rh")
        self.pts = self.pts.transpose().reshape(2, 313, 1, 1)
        self.net.getLayer(self.class8).blobs = [self.pts.astype("float32")]
        self.net.getLayer(self.conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        # Sharpening to enhance details
        self.kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])


    def enhance(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Error: Image not found or unable to load.")
            return None

        # Apply Bilateral Filter to reduce noise while keeping edges sharp
        bilateral_filter_image = cv2.bilateralFilter(image, 9, 75, 75)

        # Apply CLAHE for local contrast enhancement
        clahe_image = self.clahe.apply(bilateral_filter_image)
        sharpened_image = cv2.filter2D(clahe_image, -1, self.kernel)
        return cv2.merge([sharpened_image, sharpened_image, sharpened_image])

    def colorize_image(self, image_filename=None, cv2_frame=None):
        """
        Where all the magic happens.  Colorizes the image provided. Can colorize either
        a filename OR a cv2 frame (read from a web cam most likely)
        :param image_filename: (str) full filename to colorize
        :param cv2_frame: (cv2 frame)
        :return: Tuple[cv2 frame, cv2 frame] both non-colorized and colorized images in cv2 format as a tuple
        """
        # load the input image from disk, scale the pixel intensities to the range [0, 1], and then convert the image from the BGR to Lab color space
        image = self.enhance(image_filename) if image_filename else cv2_frame

        # image = self.enhance(image)
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        # resize the Lab image to 224x224 (the dimensions the colorization network accepts), split channels, extract the 'L' channel, and then perform mean centering
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # pass the L channel through the network which will *predict* the 'a' and 'b' channel values
        'print("[INFO] colorizing image...")'
        self.net.setInput(cv2.dnn.blobFromImage(L))
        ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))

        # resize the predicted 'ab' volume to the same dimensions as our input image
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

        # grab the 'L' channel from the *original* input image (not the resized one) and concatenate the original 'L' channel with the predicted 'ab' channels
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

        # convert the output image from the Lab color space to RGB, then clip any values that fall outside the range [0, 1]
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)

        # the current colorized image is represented as a floating point data type in the range [0, 1] -- let's convert to an unsigned 8-bit integer representation in the range [0, 255]
        colorized = (255 * colorized).astype("uint8")
        return image, colorized

    def convert_to_grayscale(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert webcam frame to grayscale
        gray_3_channels = np.zeros_like(frame)  # Convert grayscale frame (single channel) to 3 channels
        gray_3_channels[:, :, 0] = gray
        gray_3_channels[:, :, 1] = gray
        gray_3_channels[:, :, 2] = gray
        return gray_3_channels

def enhance_image(image):
    # 1. Bilateral Filter to reduce noise while keeping edges sharp
    bilateral_filter_image = cv2.bilateralFilter(image, 9, 75, 75)

    # 2. Contrast Limited Adaptive Histogram Equalization (CLAHE) for local contrast enhancement
    clahe_image = clahe.apply(bilateral_filter_image)

    # 3. Sharpening to enhance details
    kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
    sharpened_image = cv2.filter2D(clahe_image, -1, kernel)
    return sharpened_image
import cv2
import numpy as np
from imutils import paths
import imutils
import matplotlib.pyplot as plt
import sys
from PIL import Image

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def detect_blur_fft(image, size=60, thresh=10, vis=False):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    if vis:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))
        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # show our plots
        plt.show()
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)


threshold_fft = 10
vis_fft = False

image_name = "../images/4.jpg" #sys.argv[1]  # "3.jpg"
threshold_lap = 10
img = Image.open(image_name)
imgOrig_copy = img.copy()
tick = Image.open("../../../images/tick.png")
cross = Image.open("../../../images/wrong.png")

orig = cv2.imread(image_name)
orig = imutils.resize(orig, width=500)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
(mean, blurry) = detect_blur_fft(gray, size=60, thresh=threshold_fft, vis=vis_fft)
image = np.dstack([gray] * 3)
if blurry:
    color = (0, 0, 255)
    cross_copy = cross.copy()
    imgOrig_copy.paste(cross_copy, (0, 0))
    imgOrig_copy.save('Result.png', "PNG")
    imgOrig_copy.show()
else:
    color = (0, 255, 0)
    tick_copy = tick.copy()
    imgOrig_copy.paste(tick_copy, (0, 0))
    imgOrig_copy.save('Result.png', "PNG")
    imgOrig_copy.show()
text = "FFT - Blurry ({:.4f})" if blurry else "FFT - Not Blurry ({:.4f})"
text = text.format(mean)
# cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,color, 2)
print("[INFO] {}".format(text))
# show the output image
#plt.imshow("Output", image)
#plt.show()

image = orig
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fm = variance_of_laplacian(gray)
text_2 = "Laplacian - Not Blurry ({:.4f})"
if fm < threshold_lap:
    text_2 = "Laplacian - Blurry ({:.4f})"
# cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
#    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
# cv2.imshow("Image", image)
# key = cv2.waitKey(0)
text_2 = text_2.format(fm)
print("[INFO] {}".format(text_2))
# cv2.destroyAllWindows()



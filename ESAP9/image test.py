import cv2
import numpy

NETWORK_IMAGE_DIMENSIONS = (28, 28)


def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    height, width = img.shape
    img = img[0:height, int((width/2)-(height/2)):int((width/2)+(height/2))]
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = cv2.bitwise_not(img)
    return img


def process_image2(image_for_inference):
    image_for_inference=cv2.resize(image_for_inference, NETWORK_IMAGE_DIMENSIONS)
    image_for_inference = image_for_inference.astype(numpy.float32)
    image_for_inference[:] = ((image_for_inference[:])*(1.0/255.0))
    return image_for_inference


image = cv2.imread('squeeze/pupper.jpg')
image = process_image2(image)
cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

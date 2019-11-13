#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.



import numpy
import cv2
import os
import sys
#import mvnc as mvnc
import time
from mvnc import mvncapi as mvnc


from typing import List


NETWORK_IMAGE_DIMENSIONS = (28, 28)

def do_initialize():

    # Set logging level to only log errors

    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 3)
    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print('No devices found')
        return 1
    device = mvnc.Device(devices[0])
    device.open()

    graph_filename = "inference.graph"

    # Load graph file
    try :
        with open(graph_filename, mode='rb') as f:
            in_memory_graph = f.read()
    except :
        print ("Error reading graph file: " + graph_filename)
    graph = mvnc.Graph("mnist graph")
    fifo_in, fifo_out = graph.allocate_with_fifos(device, in_memory_graph)

    return device, graph, fifo_in,fifo_out


def do_inference(fifo_in,fifo_out,graph, image_for_inference, number_results):
    """ executes one inference which will determine the top classifications for an image file.

    Parameters
    ----------
    graph : Graph
        The graph to use for the inference.  This should be initialize prior to calling
    image_filename : string
        The filename (full or relative path) to use as the input for the inference.
    number_results : int
        The number of results to return, defaults to 5

    Returns
    -------
    labels : List[str]
        The top labels for the inference.  labels[i] corresponds to probabilities[i]
    probabilities: List[numpy.float16]
        The top probabilities for the inference. probabilities[i] corresponds to labels[i]
    """

    # text labels for each of the possible classfications
    labels=[ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Load tensor and get result.  This executes the inference on the NCS
    graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, image_for_inference.astype(numpy.float32), None)
    output, userobj = fifo_out.read_elem()

    # sort indices in order of highest probabilities
    five_highest_indices = (-output).argsort()[:number_results]

    # get the labels and probabilities for the top results from the inference
    inference_labels = []
    inference_probabilities = []

    for index in range(0, number_results):
        inference_probabilities.append(str(output[five_highest_indices[index]]))
        inference_labels.append(labels[five_highest_indices[index]])

    return inference_labels, inference_probabilities


def do_cleanup(device: mvnc.Device, graph: mvnc.Graph) -> None:
    """Cleans up the NCAPI resources.

    Parameters
    ----------
    device : mvncapi.Device
             Device instance that was initialized in the do_initialize method
    graph : mvncapi.Graph
            Graph instance that was initialized in the do_initialize method

    Returns
    -------
    None

    """
    graph.DeallocateGraph()
    device.CloseDevice()


def show_inference_results(image_filename, infer_labels: List[str],
                           infer_probabilities: List[numpy.float16]) -> None:

    num_results = len(infer_labels)
    for index in range(0, num_results):
        one_prediction = '  certainty ' + str(infer_probabilities[index]) + ' --> ' + "'" + infer_labels[index]+ "'"
        print(one_prediction)

    print('-----------------------------------------------------------')
from imutils.video import VideoStream

def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    height, width = img.shape
    img = img[0:height, int((width/2)-(height/2)):int((width/2)+(height/2))]
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = cv2.bitwise_not(img)
    return img

def main():
    """ Main function, return an int for program return value

    Opens device, reads graph file, runs inferences on files in digit_images
    subdirectory, prints results, closes device
    """
    vs=VideoStream(usePiCamera=True).start()
    time.sleep(1)

    # initialize the neural compute device via the NCAPI v 2
    device, graph,fifo_in,fifo_out = do_initialize()

    if (device == None or graph == None):
        print ("Could not initialize device.")
        quit(1)

    while(True):
        frame=vs.read()
        image_for_inference=frame.copy()

        # image_for_inference = cv2.cvtColor(image_for_inference, cv2.COLOR_BGR2GRAY)

        # image_for_display=image_for_inference

        image_for_display=cv2.resize(image_for_inference, (200, 200))
        process_image(image_for_inference)

        # image_for_inference=cv2.resize(image_for_inference, NETWORK_IMAGE_DIMENSIONS)
        # image_for_inference = image_for_inference.astype(numpy.float32)
        # image_for_inference[:] = ((image_for_inference[:] )*(1.0/255.0))
        cv2.imshow("mnist", image_for_display)
        cv2.waitKey(1)
        time.sleep(.25)
        infer_labels, infer_probabilities = do_inference(fifo_in, fifo_out, graph, image_for_inference, 5)
        show_inference_results(image_for_inference, infer_labels, infer_probabilities)

    # clean up the NCAPI devices
    do_cleanup(device, graph)


if __name__ == "__main__":
    sys.exit(main())


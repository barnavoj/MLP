import numpy as np
import cv2


def mlpVisualize(layers):
    width = 1500
    height = 1000
    img = np.ones((height, width, 3))
    cx = int(width / 2)
    cy = int(height / 2)
    if len(layers) > max(layers):
        node_size = width / (len(layers)) / 1.1
    else:
        node_size = height / (max(layers)) / 1.1

    node_radius = int(node_size * 0.2)
    color = (0, 0, 0)

    # input layer
    nodes = []
    xoffset = np.floor(len(layers) / 2)
    x = int(cx - xoffset * node_size)
    yoffset = int(layers[0] / 2)
    for i in range(-int(np.floor(layers[0]/2)), int(np.ceil(layers[0]/2))):
        if layers[0] % 2 == 1:
            i -= 0.5
        y = int(cy + yoffset * (i + 0.5) * node_size)
        cv2.circle(img, (x, y), node_radius, color=(0.1,0.1,0.8), thickness=-1)
        nodes.append([x, y])

    # hidden layers
    for i, layer in enumerate(layers[1:-1]):
        prevnodes = nodes[:]
        nodes = []
        i -= len(layers[1:-1])/2
        x = int(cx + (i+0.5) * node_size)
        for j in range(-int(np.floor(layer/2)), int(np.ceil(layer/2))):
            if layer % 2 == 0:
                j += 0.5
            y = int(cy + (j) * node_size)
            cv2.circle(img, (x, y), node_radius, color=color, thickness=-1)
            nodes.append([x, y])
        #draw lines from prev nodes to nodes
        for node in nodes:
            for prevnode in prevnodes:
                cv2.line(img, tuple(prevnode), tuple(node), color, thickness=2)
        

    # output layer
    prevnodes = nodes[:]
    nodes = []
    xoffset = np.floor(len(layers) / 2)
    x = int(cx + xoffset * node_size)
    for k in range(-int(np.floor(layers[-1]/2)), int(np.ceil(layers[-1]/2))):
        if layers[-1] % 2 == 1:
            k -= 0.5
        yoffset = int(layers[-1] / 2)
        y = int(cy + yoffset * (k + 0.5) * node_size)
        cv2.circle(img, (x, y), node_radius, color=(0.1,0.8,0.1), thickness=-1)
        nodes.append([x, y])
        #draw lines from prev nodes to nodes
        for node in nodes:
            for prevnode in prevnodes:
                cv2.line(img, tuple(prevnode), tuple(node), color, thickness=2)
    
    # cv2.imshow('canvas', img) 
    # cv2.waitKey(0)
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    cv2.imwrite('MLP_' + str(layers) + '.jpg', img)
    return img
    
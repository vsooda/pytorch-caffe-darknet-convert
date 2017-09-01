'''
need to delete Region layer in caffe prototxt, then we can use this code to predict boxes with caffe
torch is only use to show the result, we should remove torch code and rewrite with numpy
'''
import numpy as np
import sys,os
import cv2
import caffe
import torch
from utils import nms, get_region_boxes, load_class_names, plot_boxes
from PIL import Image
import mxnet as mx

def yolo_region(output, num_classes, anchors, num_anchors, conf_thresh = 0.3, nms_thresh = 0.6):
    output = torch.from_numpy(output).cuda()
    boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)[0]
    boxes = nms(boxes, nms_thresh)
    print 'do postprocess'
    return boxes

# set the size to 416x416
def preprocess(img, size=416):
    fill_img = np.full((size,size,3), 0.5)
    height = img.shape[0]
    width = img.shape[1]
    if size == height and size == width:
        return img, 0, 0
    scale_y = (float)(height) / size
    scale_x = (float)(width) / size
    scale = scale_x
    pad_in_width = False
    if scale_y > scale_x:
        scale = scale_y
        pad_in_width = True
    target_width = (int)(round(width / scale))
    target_height = (int)(round(height / scale))
    img = cv2.resize(img, (target_width, target_height))
    start_x = 0
    start_y = 0
    if pad_in_width:
        start_x = (size - target_width) / 2
    else:
        start_y = (size - target_height) / 2
    fill_img[start_y:start_y+target_height, start_x:start_x+target_width, :] = img[:,:,:]
    cv2.imwrite("padded.jpg", fill_img * 255)
    start_x = (float) (start_x) / size
    start_y = (float) (start_y) / size
    return fill_img, start_x, start_y

def detect(symbol, params, imgfile, num_classes, num_archors, archors):
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    img = cv2.imread(imgfile)
    # need to disable caffe->mxnet converter `first_conv`, or the channel layout will not be the same
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img, start_x, start_y = preprocess(img)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    # img = Image.open(imgfile).convert('RGB')
    # img = img.resize((416, 416), resample = Image.BILINEAR)
    # img = np.array(img, dtype=np.float32)
    # img = img / 255.0
    # img = img.transpose((2, 0, 1))

    use_caffe = False
    use_mxnet = True
    if use_caffe:
        net = caffe.Net(symbol, params, caffe.TEST)
        net.blobs['data'].data[...] = img
        out = net.forward()
        out = out['layer25-conv']
    elif use_mxnet:
        img = img[np.newaxis, :]
        sym, arg_params, aux_params = mx.model.load_checkpoint('tykk33', 0)
        mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
        mod.bind(for_training=False, data_shapes=[('data', (1, 3, 416, 416))])
        mod.set_params(arg_params, aux_params)
        from collections import namedtuple
        Batch = namedtuple('Batch', ['data'])
        test_input = mx.nd.array(img)
        input_data = Batch([test_input])
        mod.forward(input_data)
        out = mod.get_outputs()[0].asnumpy()
        print 'mxnet done'

    boxes = yolo_region(out, num_classes, anchors, num_anchors)
    class_names = load_class_names(namesfile)
    pil_image = Image.open(imgfile)

    for box in boxes:
        box[0] = (box[0] - start_x) / (1 - 2 * start_x)
        box[1] = (box[1] - start_y) / (1 - 2 * start_y)
        box[2] = box[2] / (1 - 2 * start_x)
        box[3] = box[3] / (1 - 2 * start_y)

    plot_boxes(pil_image, boxes, 'predictions.jpg', class_names)


if __name__ == "__main__":
    symbol = 'tykk3.prototxt'
    params = 'tykk3.caffemodel'
    test_image = "data/dog.jpg"
    test_image = "resize.png"
    #test_image = "data/eagle.jpg"
    anchors_str = "0.738768,0.874946,  2.42204,2.65704,  4.30971,7.04493,  10.246,4.59428,  12.6868,11.8741"
    anchors = anchors_str.split(',')
    anchors = [float(i) for i in anchors]
    for i in xrange(len(anchors)):
        print anchors[i]
    num_anchors = 5
    num_classes = 80

    detect(symbol, params, test_image, num_classes, num_anchors, anchors)




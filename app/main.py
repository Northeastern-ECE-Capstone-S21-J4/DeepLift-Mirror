import json
import trt_pose.coco
import trt_pose.models
import torch

import cv2
import torchvision.transforms as transforms
import PIL.Image

from trt_pose.parse_objects import ParseObjects
from jetcam.usb_camera import USBCamera
from jetcam.utils import bgr8_to_jpeg

from torch2trt import TRTModule

import trt_pose.plugins

import datetime as datetime
import time
import numpy as np

class DrawObjects(object):
    
    def __init__(self, topology):
        self.topology = topology
        
    def __call__(self, image, object_counts, objects, normalized_peaks, color=(0, 255, 0)):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]
        
        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
        for i in range(count):
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, color, 2)

            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), color, 2)


def load_topology(filename='human_pose.json'):
    with open('human_pose.json', 'r') as f:
        human_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    return human_pose, topology

model_trt = None
def load_model():
    global model_trt
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
    # return model_trt


def preprocess(image):
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    device = torch.device('cuda')
    image = cv2.resize(image, (224,224), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# tests if user is standing by getting the angle of the knee joints
def test_standing(keypoints):

    threshold = 5
    hip = np.array(keypoints['left_hip']).astype(float)
    knee = np.array(keypoints['left_knee']).astype(float)
    ankle = np.array(keypoints['left_ankle']).astype(float)

    thigh = hip - knee
    calf = ankle - knee

    cosine_angle = np.dot(thigh, calf) / (np.linalg.norm(thigh) * np.linalg.norm(calf))
    angle = np.arccos(cosine_angle)

    degree =  np.degrees(angle)


    print("DEGREE " + str(degree))
    return degree > 180 - threshold and degree < 180 + threshold
        

def depth_test(keypoints, squat_threshhold=.1):
    global max_thigh_length
    standing_threshold = 0.1
    result = dict()
    result['is_squat'] = False
    result['is_standing'] = False

    if float(keypoints['left_hip'][0]) == 0.0 or float(keypoints['left_knee'][0]) == 0.0 or float(keypoints['right_hip'][0]) == 0.0  or float(keypoints['right_knee'][0]) == 0.0 \
        or float(keypoints['left_ankle'][0]) == 0.0 or float(keypoints['right_ankle'][0]) == 0.0:
        return result
    elif float(keypoints['left_hip'][0]) + squat_threshhold > float(keypoints['left_knee'][0]) and float(keypoints['right_hip'][0]) + squat_threshhold > float(keypoints['right_knee'][0]):
        result['is_squat'] = True
    else:
        result['is_standing'] = test_standing(keypoints)
        
    return result


frame_num = 0
human_pose, topology = load_topology()

def print_to_file(keypoints, dump=True):
    global human_pose
    now = time.time()
    file_str = "./keypoints/" + str(now) + "_" + str(frame_num) + ".json"
    
    with open(file_str, "w") as f:
        json_keypts = {}
        for i, point in enumerate(keypoints):

            #convert to numpy to access points
            point = point.numpy()
            body_part = human_pose["keypoints"][i]
            json_keypts[body_part] = [str(point[0]), str(point[1])]

    if dump:
        json.dump(json_keypts, f, indent = 6)
    return json_keypts

def read_qr_code(image):

    qrDecoder = cv2.QRCodeDetector()
    
    # Detect and decode the qrcode
    data,bbox,rectifiedImage = qrDecoder.detectAndDecode(image)
   
    cv2.imshow('image', image)
    cv2.waitKey(1) 
    return data

# width = 224
# height= 224
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter("./output.avi", fourcc, 30, (width, height))
# parse_objects = ParseObjects(topology)
# draw_objects = DrawObjects(topology)

repcount = 0
states = dict({"prevState" : None, "currState" : None, "prePrevState" : None})
session_running = False

def execute(change):
    global frame_num 
    global topology
    global model_trt
    global repcount
    global states
    global session_running

    image = change['new']

    if session_running:

        parse_objects = ParseObjects(topology)
        draw_objects = DrawObjects(topology)


        frame_num = frame_num + 1
        
        data = preprocess(image)
        cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)

        keypoints = []
        for keypoint in peaks[0]:
            keypoints.append(keypoint[0])
            
        keypoints = print_to_file(keypoints, dump=False)
        analytics = depth_test(keypoints)

        #update states
        # states["prePrevState"] = states["prevState"]
        states["prevState"] = states["currState"]

        if analytics['is_squat']:
            states["currState"] = "squat"
            color = (0, 255, 0)
        elif analytics['is_standing']:
            states["currState"] = "standing"
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        
        if states["prevState"] == "squat" and states["currState"] == "standing":
            repcount = repcount + 1

        print(f"REPCOUNT: {repcount} Squat: {analytics['is_squat']} KP: {keypoints['left_hip'][0]},{keypoints['left_knee'][0]},{keypoints['right_hip'][0]},{keypoints['right_knee'][0]}", end='\r')
            
        draw_objects(image, counts, objects, peaks, color)
    #     image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        # image_w = bgr8_to_jpeg(image[:, ::-1, :])
        resized = cv2.resize(image[:,::-1,:], (1920, 1080), interpolation = cv2.INTER_AREA)
        # cv2.imshow('image', image[:, ::-1, :])
        cv2.imshow('image', resized)

        cv2.waitKey(1)  


        #write to video file 
    #     frame = cv2.imread(image[:, ::-1, :])
        # out.write(image[:, ::-1, :])

    else:

        data = read_qr_code(image)
        if len(data) > 0:
            session_running = True

            # do something with data here


        


# Display barcode and QR code location
def display(im, bbox):
    n = len(bbox)
    for j in range(n):
        cv2.line(im, tuple(bbox[j][0].astype(int)), tuple(bbox[ (j+1) % n][0].astype(int)), (255,0,0), 3)

    # Display results
    resized = cv2.resize(im, (1920, 1080), interpolation = cv2.INTER_AREA)
    cv2.imshow("Results", resized)
    cv2.waitKey(1) 
    


def main():

    print("Loading topology and model")
    load_model()

    WIDTH = 224 * 4
    HEIGHT = 224 * 4

    print("Starting camera")
    camera = USBCamera(width=WIDTH, height=HEIGHT, capture_fps=30, capture_device=0)

    camera.running = True

    # get qr code
    # camera.observe(read_qr_code, names='value')

    # camera.unobserve_all()

    # print("Running camera")
    camera.observe(execute, names='value')




if __name__ == '__main__':
    print("Loading")
    main()

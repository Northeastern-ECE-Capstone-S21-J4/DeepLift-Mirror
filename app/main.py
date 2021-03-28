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
import json
import requests
import boto3
import os
import sys
import copy 

from urllib3.exceptions import InsecureRequestWarning

# Suppress only the single warning from urllib3 needed.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)



api_url = 'https://api.deepliftcapstone.xyz'

class DrawObjects(object):
    """
    Draws skeleton on image based on given topology.
    """   

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

    threshold = 10

    # left
    hip = np.array(keypoints['left_hip']).astype(float)
    knee = np.array(keypoints['left_knee']).astype(float)
    ankle = np.array(keypoints['left_ankle']).astype(float)

    thigh = hip - knee
    calf = ankle - knee

    cosine_angle = np.dot(thigh, calf) / (np.linalg.norm(thigh) * np.linalg.norm(calf))
    angle = np.arccos(cosine_angle)

    left_degree =  np.degrees(angle)
    
    # right 
    hip = np.array(keypoints['right_hip']).astype(float)
    knee = np.array(keypoints['right_knee']).astype(float)
    ankle = np.array(keypoints['right_ankle']).astype(float)

    thigh = hip - knee
    calf = ankle - knee

    cosine_angle = np.dot(thigh, calf) / (np.linalg.norm(thigh) * np.linalg.norm(calf))
    angle = np.arccos(cosine_angle)
    right_degree = np.degrees(angle)

    
    # print("DEGREE " + str(degree))

    return left_degree > 180 - threshold and left_degree < 180 + threshold and right_degree > 180 - threshold and right_degree < 180 + threshold
        

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

def read_qr_code(image, qrDecoder):
    # Detect and decode the qrcode
    data,bbox,rectifiedImage = qrDecoder.detectAndDecode(image)
    resized = cv2.resize(image[:,::-1,:], (1920, 1080), interpolation = cv2.INTER_AREA)
    overlay = cv2.putText(resized, f"Please display DeepLift QR Code", org=(15,50), fontFace=1, fontScale=4, color=(255,255,255),thickness=4)
    cv2.imshow('image', overlay)
    cv2.waitKey(1) 
    return data

# def check_exit():

width = 224*4
height= 224*4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("./output.mp4", fourcc, 30, (width, height))
out_nopts = cv2.VideoWriter("./output_no_pts.mp4", fourcc, 30, (width, height))
repcount = 0
states = dict({"prevState" : None, "currState" : None, "prePrevState" : None})
session_running = False
json_data = {}
max_delta = 3.0
next_delta = time.time() + max_delta # Time that the next time endWorkout will be checked

qrDecoder = cv2.QRCodeDetector()
image_list = []

def execute(change):
    global frame_num 
    global topology
    global model_trt
    global repcount
    global states
    global session_running
    global json_data
    global max_delta
    global next_delta
    global qrDecoder
    global out
    global image_list

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
            
        blank_image = copy.deepcopy(image)
        draw_objects(image, counts, objects, peaks, color)


        overlay = cv2.putText(image, f"REPCOUNT: {repcount}", org=(15,50), fontFace=1, fontScale=4, color=(255,255,255),thickness=4)
        next_overlay = cv2.putText(overlay, f"USERNAME: {json_data['username']}, Exercise : {json_data['exerciseName']}, Weight: {json_data['weight']}", org=(15,800), fontFace=1, fontScale=1, color=(255,255,255),thickness=2)
        resized = cv2.resize(next_overlay, (1920, 1080), interpolation = cv2.INTER_AREA)
        # cv2.imshow('image', image[:, ::-1, :])
        cv2.imshow('image', resized)

        cv2.waitKey(1)  


        #write to video file 
        out.write(next_overlay)
        out_nopts.write(blank_image)

        # #check if we need to close the session and upload video data
        if time.time() > next_delta:
            url = os.path.join(api_url,'users', json_data['username'], 'lifting')
            response = requests.get(url, verify=False)
            data = json.loads(response.text)
            next_delta = time.time() + max_delta

            if not data['currentlyLifting']:
                
                # testing
                out.release()
                out_nopts.release()

                # update json with reps
                json_data["userName"] = json_data["username"]
                json_data["reps"] = repcount
                json_data["difficulty"] = data["difficulty"]
                # create workout
                url = api_url + '/workouts'

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoieWFqaW5nd2FuZzEwMjIiLCJleHBpcmVzIjoxNjE5ODQxNjAwLjB9.BtxFdI0uWoHyakfLSNm82QTQyBLX2wQhriRB6Ywb75k'

                }

                paths_response = requests.post(url, data=json.dumps(json_data), headers= headers, verify=False) # TODO: Add Bearer token in header to this request
                print(paths_response.text)
                paths_text = json.loads(paths_response.text)
                # take workout response and upload video to respective s3 bucket
                s3 = boto3.resource('s3')
                s3.meta.client.upload_file('output.mp4', 'videos-bucket-0001', paths_text['video_with_path'], ExtraArgs={'Metadata': {'ContentType': 'octet-stream'}})
                s3.meta.client.upload_file('output_no_pts.mp4', 'videos-bucket-0001', paths_text['video_without_path'], ExtraArgs={'Metadata': {'ContentType': 'octet-stream'}})
                sys.exit()

    # QR Scanning Mode
    else:
        data = read_qr_code(image, qrDecoder)
        if len(data) > 0:
            try:
                print("QR Recognized!")
                json_data = json.loads(data)
                print(data)
                session_running = True
            except:
                print("INVALID QR: Retry with QR code generated from app")
                pass

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

    # print("Running camera")'

    # spawns a thread
    camera.observe(execute, names='value')

    # time.sleep(15)
    # camera.unobserve_all()
    # out.release()



if __name__ == '__main__':
    print("Loading")
    main()

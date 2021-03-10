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


def depth_test(keypoints, squat_threshhold=.2):
    max_thigh_length = 0
    standing_threshold = 0.2
    result = dict()
    result['is_squat'] = False
    result['is_standing'] = False
    
    if float(keypoints['left_hip'][0]) == 0.0 or float(keypoints['left_knee'][0]) == 0.0 or float(keypoints['right_hip'][0]) == 0.0  or float(keypoints['right_knee'][0]) == 0.0:
        return result
    elif float(keypoints['left_hip'][0]) + squat_threshhold > float(keypoints['left_knee'][0])         and float(keypoints['right_hip'][0]) + squat_threshhold > float(keypoints['right_knee'][0]):
        result['is_squat'] = True
        
    curr_thigh_length = float(keypoints['left_knee'][0]) - float(keypoints['left_hip'][0])
    if curr_thigh_length > max_thigh_length:
        max_thigh_length = curr_thigh_length
        
    if curr_thigh_length > max_thigh_length - standing_threshold:
        result['is_standing'] = True
    
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


# width = 224
# height= 224
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter("./output.avi", fourcc, 30, (width, height))
# parse_objects = ParseObjects(topology)
# draw_objects = DrawObjects(topology)



def execute(change):
    global frame_num 
    global topology
    global model_trt
    # global parse_objects
    # global draw_objects

    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)


    frame_num = frame_num + 1
    image = change['new']
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)

    keypoints = []
    for keypoint in peaks[0]:
        keypoints.append(keypoint[0])
        
    keypoints = print_to_file(keypoints, dump=False)
    analytics = depth_test(keypoints)
    print(f"Squat: {analytics['is_squat']} KP: {keypoints['left_hip'][0]},{keypoints['left_knee'][0]},{keypoints['right_hip'][0]},{keypoints['right_knee'][0]}", end='\r')
    
    if analytics['is_squat']:
        color = (0, 255, 0)
    elif analytics['is_standing']:
        color = (0, 0, 255)
    else:
        color = (0, 255, 255)
        
    draw_objects(image, counts, objects, peaks, color)
#     image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    image_w = bgr8_to_jpeg(image[:, ::-1, :])
    
    cv2.imshow('image', image[:, ::-1, :])
    cv2.waitKey(1)  


    #write to video file 
#     frame = cv2.imread(image[:, ::-1, :])
    # out.write(image[:, ::-1, :])


def main():
    print("Loading topology and model")
    load_model()

    WIDTH = 224 * 4
    HEIGHT = 224 * 4

    print("Starting camera")
    camera = USBCamera(width=WIDTH, height=HEIGHT, capture_fps=30, capture_device=0)

    camera.running = True

    print("Running camera")
    camera.observe(execute, names='value')




if __name__ == '__main__':
    print("Loading")
    main()

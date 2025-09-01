import cv2
from matplotlib.pylab import svd
from roboflow import Roboflow
from ultralytics import YOLO
import random
import supervision as sv
import os
import time as timer
from dotenv import load_dotenv


load_dotenv()
path = "data/raw/test_video_1.MP4"
output_path = "data/labels/annotated_video.mp4"


def frame_reader(path, output_path, model):
    #grab the video based on file path
   

    cap = cv2.VideoCapture(path)
    frames = []
    time = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bounding_box_annotator = sv.RoundBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width,height))
    frame_skipper = 15 #display frame once every 15 frames

    for result in model.predict(
        path, 
        conf = .6,
        imgsz = 640,
        half = False,
        vid_stride = 5,
        stream = True,
        show = True,
        #save_frames = True,

    ): 
        print(f'we have this many detections: {len(result)}')
        for detection in result:
            box = detection.boxes
            class_index = int(box.cls)
            class_name = result.names[class_index]

            if class_name == "elixir bar":
                 print(box.xyxy[0].tolist())



        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pass
    

'''
api_key = os.getenv('API_KEY')
rf = Roboflow(api_key)
project = rf.workspace("khang-nguyen-evva6").project("elixir-timer-eh65e")
version = project.version(5)
dataset = version.download("yolov12")

model = YOLO("yolo12n.pt")
results = model.train(
    data = f"{dataset.location}/data.yaml",
    epochs = 250
)
'''

model_path = "/opt/homebrew/runs/detect/train3/weights/best.pt"
model = YOLO(model_path)

frame_reader(path, output_path, model)









        








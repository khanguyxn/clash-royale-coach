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

    
    #stop grabbing and outputting frames once you can't grab anmore
    curr_frame_index = 0
    while True:
        read, frame = cap.read()
        if read == False: 
            break
        
        curr_frame_index += 1
        #calculate current time based on what frame we're on &fps
        curr_time = curr_frame_index / fps
        time.append(curr_time)

        if curr_frame_index % frame_skipper == 0:
            result = model(
                frame, 
                imgsz = 320, 
                show_boxes = True, 
                show = True,
                conf = .6,
                )[0]
           




            #detection and annotation drawing code
            
            
            detection = sv.Detections.from_ultralytics(result).with_nms()
            annotated_image = bounding_box_annotator.annotate(
                scene = frame, 
                detections = detection)
            
            annotated_image = label_annotator.annotate(
                scene = annotated_image, detections = detection
            )
            

            
            #logic to display the cut down frames at real speed
            speed_dict = result.speed
            processing_time = sum(speed_dict.values())/1000
            print(processing_time)
            desired_time = frame_skipper/fps
            print(desired_time)
            time_skipper = int((desired_time-processing_time)*1000)
            time_skipper = abs(time_skipper)
            out.write(annotated_image)
            cv2.imshow("Annotated Video", annotated_image)
            
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    return width, height, fps, time

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









        








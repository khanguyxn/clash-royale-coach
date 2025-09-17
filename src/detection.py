import cv2
from matplotlib.pylab import svd
from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv
import pytesseract
import ai_tester
from tkinter import *
import pandas as pd
import os
from inference_sdk import InferenceHTTPClient



load_dotenv()
video_path = "data/raw/test_video_1.MP4"

def read_image(img):
     config = "--psm 13 -c tessedit_char_whitelist=0123456789:"
     #img = Image.open(img)
     img = preprocess(img)
     text = pytesseract.image_to_string(img, config = config)
     cv2.imshow("video", img)
     print(text.strip())
     return text.strip()

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)
    img = cv2.resize(img, None, fx = 5, fy = 5, interpolation=cv2.INTER_CUBIC)
    return(img)

def makeArena():
    rows = 15 +15 + 1
    cols = 18

    # Build the empty arena
    arena_grid = [["empty" for _ in range(cols)] for _ in range(rows)]

    # Mark All towers towers
    for row in range(rows):
        for col in range(cols):

            #enemy back wall
            if row == 0 and col < 6: 
                arena_grid[row][col] = "back wall"
            if row == 0 and  col > 11:
                arena_grid[row][col] = "back wall"

            #enemy king tower
            if row > 0 and row < 5 and col >6 and col < 11:
                arena_grid[row][col] = "enemy king tower"

            #enemy princess tower
            if (row > 4 and row < 8) and ((col >1 and col <5) or (col >12 and col < 16)):
                arena_grid[row][col] = "enemy princess tower"

            #river
            if (row == 15):
                arena_grid[row][col] = 'river'


            #player back wall
            if row == 30 and col < 6: 
                arena_grid[row][col] = "back wall"

            if row == 30 and  col > 11:
                arena_grid[row][col] = "back wall"

            
            #player king tower
            if row > 25 and row < 30 and col >6 and col < 11:
                arena_grid[row][col] = "player king tower"

            #enemy princess tower
            if (row > 21 and row < 25) and ((col >1 and col <5) or (col >12 and col < 16)):
                arena_grid[row][col] = "player princess tower"

    return(arena_grid)


def pixelToCoord(coord, curr_class, arena_grid):
    px, py = coord
    x1, y1 = 36, 266
    cols, rows = 18, 31
    tile_width = 815 / cols
    tile_height = 1168 / rows

    col = int((px - x1) // tile_width)
    row = int((py - y1) // tile_height)
    
    #tile width = 45.3
    #tile height = 37.68
    
    # Clamp to array bounds
    col = min(cols - 1, max(0, col))
    row = min(rows - 1, max(0, row))

    arena_grid[row][col] = curr_class

def setUpModelForCardsInPlay(client, frame):
     result = client.run_workflow(
          workspace_name="khang-nguyen-evva6",
          workflow_id="custom-workflow-2",
          images={
               "image": frame
          },
          use_cache=True # cache workflow definition for 15 minutes
     )
     return result

def analyze_cards_in_play(result):
     cards_in_play = []
     positions = []
     for index in range(len(result[0]["predictions"]["predictions"])):
          curr_prediction = result[0]["predictions"]["predictions"][index]
          curr_class = curr_prediction["class"]
          coord = int(curr_prediction['x']), int(curr_prediction['y'])
          
          cards_in_play.append(curr_class)
          positions.append(coord)
     print(f'cards is {cards_in_play} and positions are {positions}')
     return positions, cards_in_play

def draw_annotations(result, img):
     for index in range(len(result[0]["predictions"]["predictions"])):
          curr_prediction = result[0]["predictions"]["predictions"][index]
          curr_class = curr_prediction["class"]
          mid_x, mid_y = int(curr_prediction['x']), int(curr_prediction['y'])
          half_height, half_width= int(curr_prediction["height"]/2), int(curr_prediction["width"]/2)
          x1, y1 = mid_x - half_width, mid_y - half_height
          x2, y2 = mid_x + half_width, mid_y + half_height
          cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 4)
          cv2.putText(
              img,
              f'{curr_class}',
              (mid_x, y1),
              cv2.FONT_HERSHEY_SIMPLEX,
              .6,
              (0,0,255),
              4
     
          )




def scanFrame(img, hand_model, elixir_timer_model, client):
     player_hand = {
          "curr_hand_name" : [],
     } #curr hand name
     #need time, elixir, try for opponent elixir...?

     #for cards in play:
     result = setUpModelForCardsInPlay(client, img)
     arena_grid = makeArena()
     positions, cards_in_play = analyze_cards_in_play(result)
     draw_annotations(result, img)

     for index in range(len(positions)):
         pixelToCoord(positions[index], cards_in_play[index], arena_grid)

     for row in arena_grid:
         print(row)


     #for the elixir bar and timer:

     for finding in elixir_timer_model.predict(
          img,
          conf = .1,
          imgsz = (480, 864),
          show = True,
          verbose = False,

     ):
          for box in finding.boxes:
            class_index = int(box.cls)
            class_name = finding.names[class_index]

            if class_name == "elixir bar":
                 x1, y1, x2, y2 = map(int, (box.xyxy[0].tolist()))
                 elixir_region = img[y1+4:y2-30, x1+65:x2-80]
                 elixir_count = read_image(elixir_region)
                 cv2.rectangle(
                     img,
                     (x1,y1),
                     (x2, y2),
                     (0,255,0),
                     4

                 )
                 cv2.putText(
                     img,
                     class_name,
                     (x1 + 3, y1 - 10),
                     cv2.FONT_HERSHEY_COMPLEX,
                     .6,
                     (0,255,0),
                     4
                 )
                 print(f'Elixir count: {elixir_count}')
                 
            elif class_name == "timer":
                 x1, y1, x2, y2 = map(int, (box.xyxy[0].tolist()))
                 timer_region = img[y1:y2, x1:x2]
                 timer_count = read_image(timer_region)
                 cv2.rectangle(
                     img,
                     (x1,y1),
                     (x2, y2),
                     (0,255,0),
                     4

                 )
                 cv2.putText(
                     img,
                     class_name,
                     (x1 + 3, y1 - 10),
                     cv2.FONT_HERSHEY_COMPLEX,
                     .6,
                     (0,255,0),
                     4
                 )
                 print(f'timer is {timer_count}')

                

     #for the cards in hand: 
     for detection in hand_model.predict(
          img,
          conf = .2,
          imgsz = (480, 864),
          show = True,
          verbose = False,
          save = True,
          project = "data/annotated/",
     ):
        for box in detection.boxes:
            class_index = int(box.cls)
            class_name = detection.names[class_index]
            print(f"class name is: {class_name}")
            player_hand["curr_hand_name"].append(class_name)

    
     ai_tester.runModel(player_hand, arena_grid, elixir_count, timer_count)



def runVideo(video_path, hand_model, elixir_timer_model, cards_in_play_client):
    cap = cv2.VideoCapture(video_path)
    while True:
         ret, frame = cap.read()
        
         if not ret:
              break
         cv2.imshow("Video", frame)

         key = cv2.waitKey(1) & 0xFF 
         if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break       
         elif key == ord('h'):
              scanFrame(frame, hand_model, elixir_timer_model, cards_in_play_client)





'''
#Use if you haven't trained the cards in hand model yet
#https://app.roboflow.com/khang-nguyen-evva6/cr_hand_cards-fojbu/models/cr_hand_cards-fojbu-instant-2
api_key = os.getenv('API_KEY')
rf = Roboflow(api_key)
project = rf.workspace("khang-nguyen-evva6").project("cr_hand_cards-fojbu")
version = project.version(3)
dataset = version.download("yolov12")

model = YOLO("yolo12n.pt")
results = model.train(
    data = f"{dataset.location}/data.yaml",
    epochs = 10,
)
'''






def main():
     
     hand_model_path = "/opt/homebrew/runs/detect/train10/weights/best.pt"
     hand_model = YOLO(hand_model_path)

     elixir_timer_path = "/opt/homebrew/runs/detect/train3/weights/best.pt"
     elixir_timer_model = YOLO(elixir_timer_path)
     
     cards_in_play_client = client = InferenceHTTPClient(
          api_url="https://serverless.roboflow.com",
          api_key= os.getenv('API_KEY') 
     )     

     runVideo(video_path, hand_model, elixir_timer_model, cards_in_play_client)

     

if __name__ == "__main__":
     main()









        








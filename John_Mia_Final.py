import mediapipe as mp 
import time
from generate import Generate
import constants 
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
_, frm = cap.read()
height_ = frm.shape[0]
width_ = frm.shape[1]
building_image_path = 'building.png'
image_path = 'tree.png'
gen = Generate(height_, width_, building_image_path)
s_init = False
s_time = time.time()
is_game_over = False

#declarations
#hand = mp.solutions.hands
#hand_model = hand.Hands(max_num_hands=1)
#drawing = mp.solutions.drawing_utils

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Convert the frame to RGB as Face Mesh requires RGB images
rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
face_results = face_mesh.process(rgb_frame)

while True:
    ss = time.time()
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    cv2.putText(frm, "score: "+str(gen.points), (width_ - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0,0), 3)

    # Generate pipe every constants.GEN_TIME seconds
    if not(s_init):
        s_init = True 
        s_time = time.time()
    elif(time.time() - s_time) >= constants.GEN_TIME:
        s_init = False 
        gen.create()

    # Update this part: Process each frame for face landmarks
    frm.flags.writeable = False
    rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)  # Ensure this conversion is done within the loop
    face_results = face_mesh.process(rgb_frame)  # Update face_results for each frame
    frm.flags.writeable = True

    # Draw pipes & update their positions
    gen.draw_pipes(frm)
    gen.update()
    
#     gen.draw_debug_collision_boxes(frm)
    
    if face_results.multi_face_landmarks:
        for face_landmark in face_results.multi_face_landmarks:
            nose_tip = face_landmark.landmark[1]
            nose_tip_pixel = (int(nose_tip.x * width_), int(nose_tip.y * height_))

            if gen.check(nose_tip_pixel): 
                # GAME OVER
                is_game_over = True
                frm = cv2.blur(frm, (10, 10))
                frm = cv2.cvtColor(frm, cv2.COLOR_BGR2HSV)
                
                # Get the dimensions of the captured frames
                frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
                # Calculate the coordinates for the center of the screen
                center = [int(frame_width / 2), int(frame_height / 2)]
                gen.draw_image(frm, "gameover2.png", center, 700, 700)
                
                cv2.putText(frm, "GAME OVER! Press r to replay", (center[0] - 250, center[1] + 200 ), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.putText(frm, "Score : "+str(gen.points), (center[0] - 80, center[1] + 240), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)
                gen.points = 0
                
            
            # Visualize the "nose" (bird) with loaded image
            gen.draw_image(frm, "tree.png", nose_tip_pixel, 80, 80)
        

    cv2.imshow("window", frm)
    
    if is_game_over:
        key_inp = cv2.waitKey(0)
        if(key_inp == ord('r')):
            is_game_over = False 
            gen.pipes = []
            constants.SPEED = 16
            constants.GEN_TIME = 1.2
        else:
            cv2.destroyAllWindows()
            cap.release()
            break

    if cv2.waitKey(1) == 27:  # Exit on ESC
        cv2.destroyAllWindows()
        cap.release()
        break

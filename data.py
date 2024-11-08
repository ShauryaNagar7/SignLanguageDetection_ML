from function import *
from time import sleep

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        for sequence in range(no_sequences):
         for frame_num in range(sequence_length):
            # Read the image
            frame = cv2.imread(f'Image/{action}/{sequence}.png')
            if frame is None:
                print(f"Image not found: Image/{action}/{sequence}.png")
                continue

            # Make detections
            image, results = mediapipe_detection(frame, hands)
            draw_styled_landmarks(image, results)

            # Display the frame with detections
            cv2.imshow('OpenCV Feed', image)

            # Save keypoints if available
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            # Exit on pressing 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()

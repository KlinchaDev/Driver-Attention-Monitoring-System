import numpy as np
def get_look_direction(landmarks):
    # Get the key points from the facial landmarks
    left_eye = np.mean(landmarks['left_eye'], axis=0)
    right_eye = np.mean(landmarks['right_eye'], axis=0)
    nose_tip = np.mean(landmarks['nose_tip'], axis=0)
    nose_bridge = np.mean(landmarks['nose_bridge'], axis=0)

    # Calculate the midpoint between the eyes
    mid_eye = (left_eye + right_eye) / 2

    # Calculate the horizontal difference between the nose tip and the midpoint of the eyes
    horizontal_diff = nose_tip[0] - mid_eye[0]

    # Determine the look direction based on the horizontal difference
    if abs(horizontal_diff) < 5:
        return "Looking Straight"
    elif horizontal_diff < 0:
        return "Looking Left"
    else:
        return "Looking Right"
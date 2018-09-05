import win32api
import time
import cv2

from grabscreen import grab_screen

import numpy as np

WINDOW_WIDTH = win32api.GetSystemMetrics(0)
WINDOW_HEIGHT = win32api.GetSystemMetrics(1)

TRAIN_IMG_WIDTH = 300
TRAIN_IMG_HEIGHT = 300

TRAIN_IMG_X1 = int((WINDOW_WIDTH - TRAIN_IMG_WIDTH) / 2)
TRAIN_IMG_Y1 = int((WINDOW_HEIGHT - TRAIN_IMG_HEIGHT) / 2)
TRAIN_IMG_X2 = TRAIN_IMG_X1 + TRAIN_IMG_WIDTH - 1
TRAIN_IMG_Y2 = TRAIN_IMG_Y1 + TRAIN_IMG_HEIGHT - 1

def get_target_screen():
    return grab_screen(region=(TRAIN_IMG_X1,TRAIN_IMG_Y1,TRAIN_IMG_X2,TRAIN_IMG_Y2))

def get_flipped_image(img):
    return np.flip(img, axis=1)

def save_image(prefix, screen):
    screen_flip = get_flipped_image(screen)

    file_timestamp = str(time.time())
    file_name = "train_imgs/{}_{}.jpg".format(prefix, file_timestamp)
    file_name_flip = "train_imgs/{}_{}_flip.jpg".format(prefix, file_timestamp)

    cv2.imwrite(file_name, screen)
    print("Saved " + file_name)
    
    cv2.imwrite(file_name_flip, screen_flip)
    print("Saved " + file_name_flip)
            
def is_mouse_pressed():
    """Return the current state of the mouse left button."""
    return bool(win32api.GetAsyncKeyState(1))

def is_esc_pressed():
    """Return the current state of the ESC key."""
    return bool(win32api.GetAsyncKeyState(27))

if __name__ == "__main__":
    
    print("Starting...")
    time.sleep(5) #Sleep 5 seconds to get to the right window

    print("Getting training data... Press ESC to exit.")

    cnt = 0

    while not is_esc_pressed():
        cnt += 1

        delay = 0.01
        
        #If mouse is clicked, get positive train data and delay to ensure not so many data is collected
        if is_mouse_pressed():
            delay = 0.5
            
            #Get and save training data
            screen = get_target_screen()
            save_image("pressed", screen)

        #Roughly every 1 seconds, get negative train data
        elif cnt >= 100:
            cnt = 0

            screen = get_target_screen()
            save_image("not_pressed", screen)
        
        time.sleep(delay)
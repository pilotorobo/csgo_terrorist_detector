import tkinter as tk
import numpy as np
#from tkinter import *
from PIL import Image, ImageGrab, ImageTk
from io import BytesIO

import win32api

#https://stackoverflow.com/questions/40019449/python-tkinter-displaying-images-as-movie-stream

def np_to_tkimage(np_image):
    pil_image = Image.frombytes('RGB', (np_image.shape[1], np_image.shape[0]), np_image.astype('b').tostring())
    tk_image = ImageTk.PhotoImage(image = pil_image)
    return tk_image

screen_width = win32api.GetSystemMetrics(0)
screen_height = win32api.GetSystemMetrics(1)
aim_width, aim_height = 330,330

canvas_height = aim_height + 50
canvas_width = aim_width

class AimWindow:
    def __init__(self, loop_func, delay):
        root = tk.Tk()

        #Create frame
        frame = tk.Frame(root, width=canvas_width-2, height=canvas_height-2)
        frame.pack()

        #Create aim images
        aim_img_empty = np.ones((310,310,3))*255

        red_aim = np.zeros((aim_width, aim_height,3))
        red_aim[:,:,0] = np.ones((aim_width, aim_height))*255
        red_aim[10:-10, 10:-10, :] = aim_img_empty

        green_aim = np.zeros((aim_width, aim_height,3))
        green_aim[:,:,1] = np.ones((aim_width, aim_height))*220
        green_aim[10:-10, 10:-10, :] = aim_img_empty

        self.red_aim = np_to_tkimage(red_aim)
        self.green_aim = np_to_tkimage(green_aim)
        
        #Create canvas
        canvas = tk.Canvas(frame, width=canvas_width,height=canvas_height, background="white")
        canvas.place(x=-2,y=-2)
        canvas_image = canvas.create_image(0, 24, image = self.green_aim, anchor = tk.NW)
        
        target_text = canvas.create_text((3,0), text="Nothing", anchor=tk.NW, font=("Verdana", 14), fill="#00dc00")
        pred_text = canvas.create_text((3,352), text="0.000", anchor=tk.NW, font=("Verdana", 14), fill="#00dc00")

        self.root = root
        self.canvas = canvas
        self.canvas_image = canvas_image
        self.target_text = target_text
        self.pred_text = pred_text

        root.overrideredirect(True)
        root.wm_attributes("-transparentcolor", "white")
        root.wm_attributes("-topmost", 1) #Keep always on top
        root.after("idle", lambda: self._repeat(loop_func, delay))
        root.geometry("+{}+{}".format(int((screen_width-canvas_width)/2), int((screen_height-canvas_height)/2)))
        root.lift()

        root.mainloop()

    def _repeat(self, loop_func, delay):
        exit_flag = loop_func(self)
        
        if exit_flag:
            self.root.destroy()
        else:
            self.root.after(delay, lambda: self._repeat(loop_func, delay))

    def set_aim(self, aim):
        if aim == "red":
            self.canvas.itemconfig(self.canvas_image, image=self.red_aim)
            self.canvas.itemconfig(self.target_text, fill="#ff0000")
            self.canvas.itemconfig(self.pred_text, fill="#ff0000")
        else:
            self.canvas.itemconfig(self.canvas_image, image=self.green_aim)
            self.canvas.itemconfig(self.target_text, fill="#00dc00")
            self.canvas.itemconfig(self.pred_text, fill="#00dc00")

    def set_target_value(self, value):
        self.canvas.itemconfig(self.target_text, text=value)

    def set_pred_value(self, value):
        self.canvas.itemconfig(self.pred_text, text=value)

    #def set_image(self, image):
        #self.img_buffer = np_to_tkimage(image)
        #self.canvas.itemconfig(self.canvas_image, image=self.img_buffer)
    



if __name__ == "__main__":

    def loop_func(sw_self):
        aim = np.random.choice(['red', 'green'])
        pred = np.random.random()
        target = np.random.choice(['Nothing', 'Enemy'])

        sw_self.set_aim(aim)
        sw_self.set_pred_value(round(pred,3))
        sw_self.set_target_value(target)

    sw = AimWindow(loop_func, 100)

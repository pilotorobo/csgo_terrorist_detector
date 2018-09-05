from aim_window import AimWindow

from get_train_data import get_target_screen, is_esc_pressed
from train_model import CSGOModel

import cv2

if __name__ == "__main__":

    def aim_loop_func(sw_self):
        
        screen = get_target_screen()
        #Convert to jpeg quality (same used for training) use flag 1 for color images
        screen = cv2.imdecode(cv2.imencode(".jpg", screen)[1], 1) 

        pred = model.predict([screen])[0]
        #pred = 0.8

        sw_self.set_pred_value(round(pred,4))

        if pred > 0.7:
            sw_self.set_aim("red")
            sw_self.set_target_value("Enemy")
        else:
            sw_self.set_aim("green")
            sw_self.set_target_value("Nothing")

        return is_esc_pressed()

    model = CSGOModel()

    sw = AimWindow(aim_loop_func, 100)
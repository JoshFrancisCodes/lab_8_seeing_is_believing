from pupper_controller.src.pupperv2 import pupper
import math
import time
from absl import app

def run_example():
    pup = pupper.Pupper(run_on_robot=True,
                        plane_tilt=0)
    print("starting...")
    pup.slow_stand(do_sleep=True)

    yaw_rate = 0.0
    try:
        while True:
            ### TODO: Add your code here to receive the velocity command from the vision script and control the robot
            ### Read from the velocity_command file

            pup.step(action={"x_velocity": 0.0,
                                "y_velocity": 0.0,
                                "yaw_rate": yaw_rate,
                                "height": -0.14,
                                "com_x_shift": 0.005})
    finally:
        pass

def main(_):
    run_example()

app.run(main)   

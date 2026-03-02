import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import asyncio

# pipeline helper functions
from image_processing_functions import maze_image_processing
from a_star_implementation import a_star
from path_transformation import path_reduction, path_to_cm
from camera_functions import auto_run_cam2
from start_goal_detection import extract_start_goal
from find_angle import get_robot_heading_deg

# iRobot SDK functions
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, hand_over, Color, Robot, Root, Create3
from irobot_edu_sdk.music import Note

# Celebration - "Sweet Child O' Mine" by "Guns N' Roses" intro
async def celebration(robot):
    while True:
        await robot.set_wheel_speeds(25, -25)
        await robot.set_lights_spin_rgb(255, 100, 100)
        await robot.wait(3)
        await robot.play_note(Note.D5, .25)
        await robot.play_note(Note.D6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.G5, .25)
        await robot.play_note(Note.G6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.F6_SHARP, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.D5, .25)
        await robot.play_note(Note.D6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.G5, .25)
        await robot.play_note(Note.G6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.F6_SHARP, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.E5, .25)
        await robot.play_note(Note.D6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.G5, .25)
        await robot.play_note(Note.G6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.F6_SHARP, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.E5, .25)
        await robot.play_note(Note.D6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.G5, .25)
        await robot.play_note(Note.G6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.F6_SHARP, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.G5, .25)
        await robot.play_note(Note.D6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.G5, .25)
        await robot.play_note(Note.G6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.F6_SHARP, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.G5, .25)
        await robot.play_note(Note.D6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.G5, .25)
        await robot.play_note(Note.G6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.F6_SHARP, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.E6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.D6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.E6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.F6_SHARP, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.G6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.F6_SHARP, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.E6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.play_note(Note.D6, .25)
        await robot.play_note(Note.A5, .25)
        await robot.set_wheel_speeds(25, -25)
        await robot.set_lights_spin_rgb(255, 100, 100)
        await robot.wait(3)
        break


# asynchronous functions for monitoring the robot
stop_flag = asyncio.Event()

# Robot Navigation
robot = Create3(Bluetooth('Robot 6'))

# Handling bumper activation incidents
@event(robot.when_bumped, [False, True])  # Right bumper
async def handle_right_bump(robot):
    stop_flag.set()
    print("Bumped on the right side")
    await robot.set_wheel_speeds(0, 0)
    await robot.set_lights_on_rgb(255, 0, 0)
    await robot.wait(1)
    await robot.move(-8)  # Move backward
    await robot.wait(1)
    await robot.turn_left(50)  # Turn left by 50 degrees
    await robot.wait(1)
    await robot.move(4)   # Move Forward to get out of the previous path

@event(robot.when_bumped, [True, False])  # Left bumper
async def handle_left_bump(robot):
    stop_flag.set()
    print("Bumped on the left side")
    await robot.set_wheel_speeds(0, 0)
    await robot.set_lights_on_rgb(255, 0, 0)
    await robot.wait(1)
    await robot.move(-8)  # Move backward
    await robot.wait(1)
    await robot.turn_right(50)  # Turn right by 50 degrees
    await robot.wait(1)
    await robot.move(4)    # Move Forward to get out of the previous path

@event(robot.when_play)
async def play(robot):
    # Resetting the stop flag for re-activating bumper event handlers
    global stop_flag

    while True:
        await robot.set_wheel_speeds(0, 0)
        await robot.play_note(Note.A5_SHARP, 0.5)
        await robot.set_lights_on_rgb(0, 255, 0)
        stop_flag.clear()

        # Capture an image from the maze
        auto_run_cam2()

        # Reading and Transforming the image of the maze
        image_path = 'out/cam2_birdseye.jpg'
        robot_id = 13

        # Calculating the difference with initial direction in degree and make a turn
        initial_angle = get_robot_heading_deg(image_path = image_path, robot_id = robot_id)
        maze = maze_image_processing(path = image_path)

        await robot.turn_left(initial_angle)

        # Read the coordination of start and goal from the AruCo markers
        coordinates = extract_start_goal()
        start = coordinates['start_xy']
        goal = coordinates['goal_xy']

        # check if the robot reached the goal
        if goal == None:
            print("Awesome! Goal is Achieved!")
            break

        print(f'{type(start)}, {start}')
        print(f'{type(goal)}, {goal}')

        safety_margin = 100
        path = a_star(maze, start, goal, clearance_px = safety_margin, turn_penalty = 0.3)

        # Reducing the initial safety margin by 5 pixels until a valid path is generated
        while not path:
                safety_margin -= 5
                print(f'Still searching with safety margin of {safety_margin} pixels')
                path = a_star(maze, start, goal, clearance_px = safety_margin, turn_penalty = 0.3)

        print(f'Found the path with safety margin: {safety_margin} pixels')

        # Reduce and translate the path to polar coordinates (angle, magnitude)
        path_reduced = path_reduction(path)
        path_cm = path_to_cm(path_reduced, maze, (474, 417))

        await robot.set_lights_on_rgb(0, 255, 100)
        await robot.play_note(Note.C5_SHARP, 1.5)

        for angle, magnitude in path_cm:
            if stop_flag.is_set():
                break

            # Compensating the internal error of the robot
            magnitude *= 0.93
            angle *= 0.995

            await robot.turn_right(float(angle))
            await robot.move(magnitude)

        await robot.set_wheel_speeds(0, 0)
        await robot.wait(15)

    await celebration(robot)

robot.play()

import numpy as np
import sys

from gym import utils
from gym import Env, spaces
from gym.utils import seeding

import math
from math import pi as PI
import random

all_envs = []

class CrawlingRobotEnv(Env):

    def close_gui(self):
        if self.root is not None:
            self.root.destroy()
            self.root = None

    def __init__(self, horizon=np.inf, render=False):
        if render:
            import tkinter
            for env in all_envs:
                env.close_gui()
            #all_envs.clear()
            all_envs.append(self)
            root = tkinter.Tk()
            root.title('Crawler GUI')
            root.resizable(0, 0)
            self.root = root
            canvas = tkinter.Canvas(root, height=500, width=1100)
            canvas.grid(row=2, columnspan=10)

            def close():
                if self.root is not None:
                    self.root.destroy()
                    self.root = None
            root.protocol('WM_DELETE_WINDOW', lambda: close)
            root.lift()
            root.attributes('-topmost', True)
            import platform, subprocess, os
            if platform.system() == 'Darwin':
                tmpl = 'tell application "System Events" to set frontmost of every process whose unix id is {} to true'
                script = tmpl.format(os.getpid())
                output = subprocess.check_call(['/usr/bin/osascript', '-e', script])
            root.attributes('-topmost', False)
        else:
            canvas = None
            self.root = None
            
        robot = CrawlingRobot(canvas)
        self.crawlingRobot = robot

        self._stepCount = 0
        self.horizon = horizon

        self.state = None

        self.nArmStates = 9
        self.nHandStates = 13

        # create a list of arm buckets and hand buckets to
        # discretize the state space
        minArmAngle, maxArmAngle = self.crawlingRobot.getMinAndMaxArmAngles()
        minHandAngle, maxHandAngle = self.crawlingRobot.getMinAndMaxHandAngles()
        armIncrement = (maxArmAngle - minArmAngle) / (self.nArmStates-1)
        handIncrement = (maxHandAngle - minHandAngle) / (self.nHandStates-1)
        self.armBuckets = [minArmAngle+(armIncrement*i) \
                           for i in range(self.nArmStates)]
        self.handBuckets = [minHandAngle+(handIncrement*i) \
                            for i in range(self.nHandStates)]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple(
            [spaces.Discrete(self.nArmStates), spaces.Discrete(self.nHandStates)]
        )

        # Reset
        self.reset()

    @property
    def stepCount(self):
        return self._stepCount

    @stepCount.setter
    def stepCount(self, val):
        self._stepCount = val
        self.crawlingRobot.draw(val, self.root)

    


    def reset(self):
        """
         Resets the Environment to the initial state
        """
        ## Initialize the state to be the middle
        ## value for each parameter e.g. if there are 13 and 19
        ## buckets for the arm and hand parameters, then the intial
        ## state should be (6,9)
        ##
        ## Also call self.crawlingRobot.setAngles()
        ## to the initial arm and hand angle

        armState = self.nArmStates // 2
        handState = self.nHandStates // 2
        self.state = armState, handState
        self.crawlingRobot.setAngles(self.armBuckets[armState], self.handBuckets[handState])
        self.crawlingRobot.positions = [20, self.crawlingRobot.getRobotPosition()[0]]

        self.stepCount = 0

class CrawlingRobot:

    def __init__(self, canvas):

        ## Canvas ##
        self.canvas = canvas
        self.velAvg = 0
        self.lastStep = 0

        ## Arm and Hand Degrees ##
        self.armAngle = self.oldArmDegree = 0.0
        self.handAngle = self.oldHandDegree = -PI/6

        self.maxArmAngle = PI/6
        self.minArmAngle = -PI/6

        self.maxHandAngle = 0
        self.minHandAngle = -(5.0/6.0) * PI

        self.robotWidth = 150
        self.robotHeight = 40
        self.armLength = 60
        self.handLength = 40
        self.positions = [0,0]

        ## Draw Ground ##
        if canvas is not None:
            self.totWidth = canvas.winfo_reqwidth()
            self.totHeight = canvas.winfo_reqheight()
            self.groundHeight = 90
            self.groundY = self.totHeight - self.groundHeight

            self.ground = canvas.create_rectangle(
                0,
                self.groundY,self.totWidth,self.totHeight, fill='green'
            )

            ## Robot Body ##
            self.robotPos = (self.totWidth / 5 * 2, self.groundY)
            self.robotBody = canvas.create_polygon(0,0,0,0,0,0,0,0,0,0, fill='black')

            ## Robot Arm ##
            self.robotArm = canvas.create_line(0,0,0,0,fill='orange',width=5)

            ## Robot Hand ##
            self.robotHand = canvas.create_line(0,0,0,0,fill='red',width=3)

            # canvas.focus_force()

        else:
            self.robotPos = (20, 0)

    def setAngles(self, armAngle, handAngle):
        """
            set the robot's arm and hand angles
            to the passed in values
        """
        self.armAngle = armAngle
        self.handAngle = handAngle

    def getAngles(self):
        """
            returns the pair of (armAngle, handAngle)
        """
        return self.armAngle, self.handAngle

    def getRobotPosition(self):
        """
            returns the (x,y) coordinates
            of the lower-left point of the
            robot
        """
        return self.robotPos

    

    def getMinAndMaxArmAngles(self):
        """
            get the lower- and upper- bound
            for the arm angles returns (min,max) pair
        """
        return self.minArmAngle, self.maxArmAngle

    def getMinAndMaxHandAngles(self):
        """
            get the lower- and upper- bound
            for the hand angles returns (min,max) pair
        """
        return self.minHandAngle, self.maxHandAngle

    def getRotationAngle(self):
        """
            get the current angle the
            robot body is rotated off the ground
        """
        armCos, armSin = self.__getCosAndSin(self.armAngle)
        handCos, handSin = self.__getCosAndSin(self.handAngle)
        x = self.armLength * armCos + self.handLength * handCos + self.robotWidth
        y = self.armLength * armSin + self.handLength * handSin + self.robotHeight
        if y < 0:
            return math.atan(-y/x)
        return 0.0


    ## You shouldn't need methods below here


    def __getCosAndSin(self, angle):
        return math.cos(angle), math.sin(angle)

    

    def draw(self, stepCount, root):
        if self.canvas is None or root is None:
            return
        x1, y1 = self.getRobotPosition()
        x1 = x1 % self.totWidth

        ## Check Lower Still on the ground
        if y1 != self.groundY:
            raise 'Flying Robot!!'

        rotationAngle = self.getRotationAngle()
        cosRot, sinRot = self.__getCosAndSin(rotationAngle)

        x2 = x1 + self.robotWidth * cosRot
        y2 = y1 - self.robotWidth * sinRot

        x3 = x1 - self.robotHeight * sinRot
        y3 = y1 - self.robotHeight * cosRot

        x4 = x3 + cosRot*self.robotWidth
        y4 = y3 - sinRot*self.robotWidth

        self.canvas.coords(self.robotBody,x1,y1,x2,y2,x4,y4,x3,y3)

        armCos, armSin = self.__getCosAndSin(rotationAngle+self.armAngle)
        xArm = x4 + self.armLength * armCos
        yArm = y4 - self.armLength * armSin

        self.canvas.coords(self.robotArm,x4,y4,xArm,yArm)

        handCos, handSin = self.__getCosAndSin(self.handAngle+rotationAngle)
        xHand = xArm + self.handLength * handCos
        yHand = yArm - self.handLength * handSin

        self.canvas.coords(self.robotHand,xArm,yArm,xHand,yHand)
        root.update()
    #        self.lastVel = velocity

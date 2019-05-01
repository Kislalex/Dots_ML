import sys
import random
import math
import time

import numpy as np
from smart import Brain

def getAccFromAction(action):
    return 4 * action

def newVelocity(vel, acc, ttl, max_vel):
    vel = np.add(vel, ttl * acc)
    if (np.linalg.norm(vel) > max_vel):
            vel = vel * (max_vel / np.linalg.norm(vel))
    return vel

class Dot:
    vision_depth = 50
    max_vel = 10
    def __init__(self, start, rule):
        self.rule = tuple(rule)
        # create the smart Brain
        self.dot_brain = Brain(rule)
        #
        self.pos = np.array(start)
        self.vel = np.array([0, 0])
        self.acc = np.array([0, 0])
        #
        self.dead = False
        self.reached_goal = False
        self.checkpoints = 0
        self.min_distance = 2000
        #
        self.time = 0
        self.score = 0

    def move(self, field, ttl):
        # move only if alive
        if (self.dead or self.reached_goal):
            return
        #check the objectives
        brain_info = field.gatherInfo(self.pos, self.vel, self.checkpoints)
        self.min_distance = min(self.min_distance, brain_info[0] * 2000)
        # get the brain signal
        action = self.dot_brain.signal(brain_info)
        # finally move, limiting velocity
        self.acc = getAccFromAction(action)
        self.vel = newVelocity(self.vel, self.acc, ttl, self.max_vel)
        self.pos = np.add(self.pos, ttl * self.vel)
        self.time = self.time + 1
        

    def update(self, field):
        if ((not self.dead) and (not self.reached_goal)):
            self.move(field, 1)
            if (field.isCloseToObsticale(self.pos)):
                self.dead = True
            if (field.isFinished(self.pos)):
                self.reached_goal = True
                return True
            if (not self.reached_goal):
                if (field.isPassingCheckPoint(self.pos, self.checkpoints)):
                    self.checkpoints += 1
        return False

    def computeScore(self, field):
        self.score = (4 ** self.checkpoints)
        if (self.reached_goal):
            self.score *= (1 + 100.0 / self.time)
        else:
            distace_to_score = self.min_distance + 10
            self.score *= (1.0 / (distace_to_score ** 2))
            
    def reproduce(self, start):
        baby = Dot(start, self.rule)
        baby.dot_brain = self.dot_brain.copy()
        return baby

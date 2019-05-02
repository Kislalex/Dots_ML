import sys
import random
import time

import numpy as np
from brain import Brain

def getAccFromAction(action):
    return action

def newVelocity(vel, acc, ttl, max_vel):
    vel = np.add(vel, ttl * acc)
    if (np.linalg.norm(vel) > max_vel):
            vel = vel * (max_vel / np.linalg.norm(vel))
    return vel

def scoreForPoint(next_checkpoints, 
                  max_checkpoint, 
                  distace_to_goal, 
                  distace_to_checkpoint,
                  min_distance,
                  time):
    score = 0
    if (next_checkpoints > max_checkpoint):
        score = (100 + 10000.0 / time)
    else:
        if (max_checkpoint == 0):
            score = (100.0 / ((distace_to_goal + min_distance) ** 2))
        else:
            mult = (2 ** next_checkpoints)
            dist_to_score = (distace_to_checkpoint + 10)
            score = 1 * next_checkpoints + (10.0 / (distace_to_checkpoint ** 2))
            score *= mult
    return score
    

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
        self.min_distance = 5000
        #
        self.time = 0
        self.score = 0

    def move(self, field, ttl):
        # move only if alive
        if (self.dead or self.reached_goal):
            return
        #check the objectives
        brain_info = field.gatherInfo(self.pos, self.vel, self.checkpoints)
        # get the brain signal
        action = self.dot_brain.signal(brain_info)
        # finally move, limiting velocity
        self.acc = getAccFromAction(action)
        self.vel = newVelocity(self.vel, self.acc, ttl, self.max_vel)
        self.pos = np.add(self.pos, ttl * self.vel)
        self.time = self.time + 1
        #Update min distance by looking at the goal TODO change to the distance to goal not to 1000 checkpoint
        self.min_distance = min(self.min_distance, field.findNextGoal(self.pos, self.vel, 1000)[0] * 2000)

    def update(self, field):
        if ((not self.dead) and (not self.reached_goal)):
            self.move(field, 1)
            if (field.isCloseToObsticale(self.pos)):
                self.dead = True
            if (field.isFinished(self.pos)):
                self.reached_goal = True
                self.checkpoints += 1
                return True
            if (not self.reached_goal):
                if (field.isPassingCheckPoint(self.pos, self.checkpoints)):
                    self.checkpoints += 1
        return False

    def computeScore(self, field):
        closest_goal = field.findNextGoal(self.pos, self.vel, self.checkpoints)[0] * 2000
        distace_to_goal = field.findNextGoal(self.pos, self.vel, 1000)[0] * 2000
        max_checkpoint = len(field.checkpoints)
        self.score = scoreForPoint(self.checkpoints, 
                                   max_checkpoint, 
                                   distace_to_goal, 
                                   closest_goal, 
                                   self.min_distance,
                                   self.time)
            
    def reproduce(self, start):
        baby = Dot(start, self.rule)
        baby.dot_brain = self.dot_brain.copy()
        return baby

import sys
import random
import time

import numpy as np
from NeuralNetwork.brain import Brain
from field import turn_unit_vector_by_angle as turn_by_angle


def get_score_from_dist(dist_, highscore):
    dist_ *= highscore / (2 * 1000)
    if dist_ > highscore / 2:
        dist_ = highscore / 2
    return highscore / 2 - dist_


def get_acc_from_action(action, vel):
    if np.linalg.norm(vel) < 0.001:
        return np.array([1, -1])
    alpha = action[0] * 2 * np.pi
    value = 2 * (action[1] + 1)
    acc = turn_by_angle(vel, alpha)
    return acc * value


def update_velocity(vel, acc, ttl, max_vel):
    vel = vel + ttl * acc
    if np.linalg.norm(vel) > max_vel:
        vel = vel * (max_vel / np.linalg.norm(vel))
    return vel


def compute_score_for_point(
    checkpoints,
    max_checkpoint,
    min_distance,
    time,
):
    score = 0
    if checkpoints > max_checkpoint:
        score = 100.0 + 100.0 / time
    else:
        checkpoint_price = 100.0 / (max_checkpoint + 1)
        score = checkpoint_price * checkpoints
        score += get_score_from_dist(min_distance, checkpoint_price)
    return score


class Dot:
    vision_depth = 50
    max_vel = 10

    def __init__(self, start, rule):
        self.rule = tuple(rule)
        # create the smart Brain
        self.dot_brain = Brain()
        for i in range(len(rule) - 1):
            self.dot_brain.add_layer(0, 1, 1, 1, (rule[i], rule[i + 1]))
        self.dot_brain.mutate(1.0, 1.0)
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
        if self.dead or self.reached_goal:
            return
        # check the objectives
        field_info = field.gather_info(self.pos, self.vel, self.checkpoints)
        # get the brain signal
        action = self.dot_brain.compute(field_info)[0]
        # finally move, limiting velocity
        self.acc = get_acc_from_action(action, self.vel)
        self.vel = update_velocity(self.vel, self.acc, ttl, self.max_vel)
        self.pos = np.add(self.pos, ttl * self.vel)
        self.time = self.time + 1
        # Update min distance by looking at the goal TODO change to the distance to goal not to 1000 checkpoint
        self.min_distance = min(
            self.min_distance,
            field.find_next_goal(self.pos, self.vel, 1000)[0] * 1000.0,
        )

    def update(self, field):
        if (not self.dead) and (not self.reached_goal):
            self.move(field, 1)
            if field.is_close_to_obsticale(self.pos):
                self.dead = True
            if self.checkpoints >= len(field.checkpoints):
                if field.is_finished(self.pos):
                    self.reached_goal = True
                    self.checkpoints += 1
                    return True
            if not self.reached_goal:
                if field.is_passing_checkpoint(self.pos, self.checkpoints):
                    self.checkpoints += 1
        return False

    def update_score(self, field):
        closest_goal = (
            field.find_next_goal(self.pos, self.vel, self.checkpoints)[0] * 2000
        )
        distace_to_goal = field.find_next_goal(self.pos, self.vel, 1000)[0] * 2000
        max_checkpoint = len(field.checkpoints)
        self.score += compute_score_for_point(
            self.checkpoints,
            max_checkpoint,
            self.min_distance,
            self.time,
        )

    def clear_score(self):
        self.score = 0

    def reproduce(self, start):
        baby = Dot(start, self.rule)
        baby.dot_brain = self.dot_brain.copy()
        return baby

#!/usr/bin/python3.7
import cv2
import os
import time
import math

from population import Population
from brain import Brain
from field import Field

import numpy as np

height = 900
width = 1200
blue = (255,0,0)
orange = (0,165,255)
green = (0,128,0)
red = (0, 0, 255)
yellow = (140,230,240)
ttl = 1
directions = list(np.linspace(-math.pi / 2, math.pi / 2, 9 - 2))
     
def turnByAngle(vect, angle):
    turn = np.array([[ np.cos(angle),  np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]])
    res = np.dot(turn, vect)
    res = np.divide(res, np.linalg.norm(res))
    return res

def createFields(num):
    fields = []
    s='fields/field'
    n = 0
    while (os.path.isfile(s + str(n) + '.txt')):
        f = open(s + str(n) + '.txt',"r")
        finish_ = tuple(map(int,f.readline().split(' ')))
        field = Field(finish_, num)
        while True:
            line = list(map(int,f.readline().split(' ')))
            polygon = list(zip(line[::2], line[1::2]))
            if (len(polygon) < 2):
                break
            field.addObsticale(polygon)
        while True:
            line = list(map(int,f.readline().split(' ')))
            polygon = list(zip(line[::2], line[1::2]))
            if (len(polygon) < 2):
                break
            field.addCheckPoint(polygon)
        n = n + 1
        fields.append(field)
    return fields


def drawLeaderInfo(img, leader, field):
    cv2.circle(img, tuple(leader.pos.astype(int)), 5, green, -1)
    cv2.circle(img, tuple(leader.pos.astype(int)), 5, (0,0,0), 1)
    if(np.linalg.norm(leader.vel) > 0.001): 
        data = field.gatherInfo(leader.pos, leader.vel, leader.checkpoints)
        alpha = data[1] * 10.0
        dist_ = data[0] * 2000
        dir_ = turnByAngle(leader.vel, alpha)
        goal = np.add(leader.pos, dist_ * dir_)
        cv2.circle(img, tuple(goal.astype(int)), 5, orange, -1)
        for i in range(len(directions)):
            alpha = directions[i]
            dist_ = data[i+2] * 100
            dir_ = turnByAngle(leader.vel, alpha)
            goal = np.add(leader.pos, dist_ * dir_)
            cv2.line(img, tuple(leader.pos.astype(int)), 
                tuple(goal.astype(int)), (0,0,0), 1)
            cv2.circle(img, tuple(goal.astype(int)), 3, orange, -1)
            

def redraw(blank, fields, population):
    n = 0
    m = 0
    while True:
        x = population.allDotsStopped()
        if (x == 0):
            if (population.dots[0].reached_goal):
                m += 1
                if (m > 10):
                    m = 0
                    n += 1
                    if (n >= len(fields)):
                        n = 0
                    print(n)
            population.computeScore(fields[n])
            population.naturalSelection()
            population.mutation()
        else:
            begin = time.time()
            population.update(fields[n])
            end = time.time()
            img = blank.copy()
            for polygon in fields[n].obsticales:
                cv2.polylines(img, [np.array(polygon.exterior.coords, np.int32)], True, blue, 5)
            for polygon in fields[n].checkpoints:
                cv2.polylines(img, [np.array(polygon.exterior.coords, np.int32)], True, yellow, 1)
                
            cv2.circle(img, tuple(fields[n].finish.astype(int)), 10, red, -1)
            cv2.circle(img, tuple(fields[n].finish.astype(int)), 10, (0,0,0), 1)

            for dot in population.dots[1:]:
                cv2.circle(img, tuple(dot.pos.astype(int)), 3, (0,0,0), -1)
            
            drawLeaderInfo(img, population.dots[0], fields[n])
            
            cv2.putText(img, "gen : " + str(population.gen), (10,30), 2, 0.6, (0,0,0))
            cv2.putText(img, "fps : " + str(1.0 / (end - begin)), (10,50), 2, 0.6, (0,0,0))
            cv2.putText(img, "alive : " + str(x), (10,70), 2, 0.6, (0,0,0))
            cv2.putText(img, "level : " + str(n) + ',' + str(m), (10,90), 2, 0.6, (0,0,0))
            cv2.imshow('ML',img)
            key = cv2.waitKey(ttl)
            if (key == ord('q')):
                break

blank = 255 * np.ones((height, width, 3),dtype = "uint8")
cv2.imshow('ML',blank)

rule = (9,50,12,50,10,2)

fields = createFields(rule[0])
population = Population(np.array([50, height - 50]), 80, rule)
redraw(blank, fields, population)

cv2.destroyAllWindows()

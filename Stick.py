from Ball import *
from globals import *

#from Stick import Stick
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Stick(object):
    def __init__(self, left, top, ball):
        self.left = left
        self.right= self.left+ST_W
        self.top  = top
        self.bot  = self.top + ST_H
        self.center = Point(self.top+ST_H/2, self.left+ST_W/2)
        self.pos = self.center.x
        self.endless_x_point = 0
        self.ball = ball

    def update(self):
        self.right= self.left+ ST_W
        self.bot  = self.top + ST_H
        self.center.x = self.top  + ST_H/2
        self.center.y = self.left + ST_W/2
    def predict_pos(self, ball, border_w, real_screen_h): # Явный предикционный алгоритм движения
        # BA_SPEED[0] -- скорость по вертикали; BA_SPEED[1] -- скорость по горизонтали
        # center.x -- коорд по вертикали (выше - меньше), center.y -- коорд по горизонтали
        if ball.vel.x == 0: return ball.center.y
        dist1 = abs(self.center.y - ball.center.y)

        dist_in_frames = dist1 / abs(ball.vel.y)

        self.endless_x_point = ball.center.x + ball.vel.x * dist_in_frames - 2 * border_w # 10 - координата начала поля

        self.pos = min ((self.endless_x_point) % (2* real_screen_h) + border_w, 2*real_screen_h - (self.endless_x_point) % (2* real_screen_h) + border_w)
        return self.pos

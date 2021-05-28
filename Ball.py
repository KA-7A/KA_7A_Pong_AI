import functions
import math

#       |======|
#
#
#
#  Y    x_vel
# /|\    |
#  |     <- o
#  |       /|
#  |      / |
#  |     /  V - y_vel
#  |
#  |        |======|
#  0--------------------------->X
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Ball(object):
    def __init__(self, left, top, BA_S, BA_SPEED):
        self.left = left
        self.right= self.left+ BA_S
        self.top  = top
        self.bot  = self.top + BA_S
        self.center = Point(self.top+BA_S/2,self.left+BA_S/2)
        self.BA_S = BA_S
        self.vel = Point(BA_SPEED[0], BA_SPEED[1])
        self.absvel = ((self.vel.x)**2 + (self.vel.y)**2)**0.5
        self.i = 0
        self.MAX_POW = 1
        self.POW = 1
        self.dP = 0.2
        self.dS = math.pi # Попытались взять иррациональное число, чтобы не получился в какой-то момент ноль, ага угу
        pass
    def update(self):
        self.right= self.left+ self.BA_S
        self.bot  = self.top + self.BA_S
        self.center.x = self.top  + self.BA_S/2
        self.center.y = self.left + self.BA_S/2

    def moving(self, st1, st2, border_w, real_screen_h):
        if self.top <=border_w or self.top >= real_screen_h - 10:
            self.vel.x *= -1
        # С палкой
        if self.left <= st1.right: # С левой
            #print()
            if st1.top <= self.bot and self.top <= st1.bot: # Если столкновение произошло
                #print("POW = ", self.POW)
                pow = 1
                if self.POW < self.MAX_POW:
                    pow = self.POW
                    self.POW += self.dP
                self.vel.y *= -pow
                if st1.dir == "up":
                    self.vel.x += self.dS
                if st1.dir == "down":
                    self.vel.x -= self.dS
            else:
                #print("Левый, b.c, st1.c, st1.pos ->", self.center.x, st1.center.x, st1.pos)
                self.top = real_screen_h//2
                self.left= real_screen_h//2
                #i = 0
            functions.predict(st1, st2, self, border_w, real_screen_h)
        if self.right >= st2.left:
            #print()
            if st2.top <= self.bot and self.top <= st2.bot:
                #print("POW = ", self.POW)
                pow = 1
                if self.POW < self.MAX_POW:
                    pow = self.POW
                    self.POW += self.dP
                self.vel.y *= -pow
                if st2.dir == "up":
                    self.vel.x += self.dS
                if st2.dir == "down":
                    self.vel.x -= self.dS
            else:
                #print("Правый, b.c, st2.c, st2.pos ->", self.center.x, st2.center.x, st2.pos)
                self.top = real_screen_h//2
                self.left= real_screen_h//2
                #i = 0
            functions.predict(st1, st2, self, border_w, real_screen_h)
    # Движение шара
        ##print("ball_moving: update")
        self.left += self.vel.y
        self.top  += self.vel.x
        #print(self.vel.x, self.vel.y)



    ##print("ball_moving: end")

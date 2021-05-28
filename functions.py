#импорт
import os
import pygame
from pygame.locals import *
import sys
from globals import *
# dcree heigh/width
SC_W = 800
SC_H = 400

BORDER = 1
FPS = 30

# stick width/ height, ball size
ST_H = 70
ST_W = 10
BA_S = 10

# colors
BLACK = (0  , 0  , 0  )
WHITE = (255, 255, 255)
BLUE  = (0  , 0  , 200)
RED   = (200, 0  , 0  )
GREEN = (0  , 200, 0  )
YELLOW= (255, 255, 0  )

ST_SPEED = 15

dP = 0.2
STOP = "stop"
UP = "up"
DOWN = "down"


def drawing(win, st1, st2, ball):
    # Границы
    pygame.draw.rect(win, WHITE, (0, 0, SC_W, BORDER))
    pygame.draw.rect(win, WHITE, (0, 0, BORDER, SC_H))
    pygame.draw.rect(win, WHITE, (SC_W-BORDER, 0, BORDER, SC_H))
    pygame.draw.rect(win, WHITE, (0, SC_H-BORDER, SC_W, BORDER))

    pygame.draw.rect(win, WHITE, (st1.left, st1.top, ST_W, ST_H))   # stick left
    pygame.draw.rect(win, WHITE, (st2.left, st2.top, ST_W, ST_H))   # stick right
    pygame.draw.rect(win, WHITE, (ball.left, ball.top, BA_S, BA_S)) # ball

    if NN_on:
        for i in range(len(st1.NN.layer_list)):
            for j in range(len(st1.NN.layer_list[i])):
                if not i == len(st1.NN.layer_list) -1 :
                    if st1.NN.layer_list[i][j] > 0:
                        color = GREEN
                    else:
                        color = RED
                    pygame.draw.circle(win, color, (i*40 + SC_W + 50, j*40 + 20), 5)
                    if i < len(st1.NN.weights_list) - 1 :
                        for k in range(len(st1.NN.weights_list[i+1])):
                            if st1.NN.weights_list[i][j][k] > 0: color = GREEN
                            else : color = RED
                            pygame.draw.aalines(win, color, True, [(i*40 + SC_W + 50, j*40 + 20),((i+1)*40 + SC_W + 50, k*40 + 20)])
                else:
                    if st1.NN.layer_list[i][j] > 0:
                        color = GREEN
                    else:
                        color = RED
                    pygame.draw.circle(win, color, (i*40 + SC_W + 50, j*40 + 20), 5)
                    break
        pygame.display.update()


def on_keys(st1, st2):
    for i in pygame.event.get():
        if i.type == pygame.QUIT:        os._exit(os.EX_OK)
        elif i.type == pygame.KEYDOWN:
            if i.key == pygame.K_UP:     st2.dir = UP
            elif i.key == pygame.K_DOWN: st2.dir = DOWN
            elif i.key == pygame.K_w:    st1.dir = UP
            elif i.key == pygame.K_s:    st1.dir = DOWN
        elif i.type == pygame.KEYUP:
            if i.key in [pygame.K_UP, pygame.K_DOWN]:
                st2.dir = STOP
            elif i.key in [pygame.K_w, pygame.K_s]:
                st1.dir = STOP


def predict(st1, st2, ball, border_w, real_screen_h):
    ball.update()
    st1.update()
    st2.update()

    st1.predict_pos(ball, border_w, real_screen_h)
    st2.predict_pos(ball, border_w, real_screen_h)


def stick_moving(st1, st2):
    if st2.dir == UP:
        if st2.top > BORDER        : st2.top -= ST_SPEED
    elif st2.dir == DOWN:
        if st2.bot < SC_H - BORDER : st2.top += ST_SPEED
    if st1.dir == UP:
        if st1.top > BORDER        : st1.top -= ST_SPEED
    elif st1.dir == DOWN:
        if st1.bot < SC_H - BORDER : st1.top += ST_SPEED


def update(st1, st2, ball):
    st1.update()
    st2.update()
    ball.update()


# Реакционный алгоритм
    #print(st1.bot, st1.pos, st1.top)
    #if st1.top <= st1.pos <= st1.bot and start:            # Если шар не в палке
        #print("stick is in pos")
    #    if st1.center_x <= st1.pos:              # Если ниже
    #        st1.dir = UP
    #    if st1.center_x >= st1.pos:              # Если выше
    ##        st1.dir = DOWN
    #else:st1.dir = STOP

# Ответка на предикционный
    #if st2.bot < st2_pos: ST2_DIR = DOWN
    #elif st2.top > st2_pos: ST2_DIR = UP
    #else: ST2_DIR = STOP

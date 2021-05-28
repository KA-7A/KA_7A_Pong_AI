#импорт
from stick import Stick
import pygame
from pygame.locals import *
import sys
from NeuralNetwork import *
import functions
from Ball import Ball
import ai_pols
from globals import *
# player: 1 - есть игрок, 0 - явный алгоритм, -1 - Нейросеть (Keras), 1 - нейросеть ручная
if True:
    player1 = 1
    player2 = 0

    #stick width/ height, ball size
    ST_H = 70
    ST_W = 10
    BA_S = 10

    #dcree heigh/width
    SC_W = 800
    SC_H = 400
    NN_WIDTH = 200

    border_w = BORDER = 1
    real_screen_h = R_SC_H = 0
    FPS = 60

    #colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    #ball's and stick's speed
    # BA_SPEED[0] -- скорость по вертикали; BA_SPEED[1] -- скорость по горизонтали
    X = 4
    Y = 4
    BA_SPEED = [0, 0]
    ST_SPEED = 15

    #consts
    right = 0
    top = 0
    STOP = "stop"
    UP = "up"
    DOWN = "down"
    i = 1

    st1_pos = 0
    st2_pos = 0

    min_top = SC_H
    max_bot = 0
    min_top_pr = SC_H+1
    max_bot_pr = -1

#инициализация
pygame.init()
# ai_pols.ai_init()  # Инициализируем нейросеть и обучаем её

win = pygame.display.set_mode((SC_W + NN_WIDTH, SC_H))
pygame.display.update()
clock = pygame.time.Clock()

ball = Ball(SC_W // 2, SC_H // 2, BA_S, BA_SPEED)
st1 = Stick(SC_W * 1  / 60,  BORDER, ball)
st2 = Stick(SC_W * 59 / 60-ST_W, 40, ball)

#главный цикл
pl = True
st1.dir = STOP
st2.dir = STOP
start = 1       # Переменная для того, чтобы определить размер экрана (да-да, такой вот ерундой занимаемся)
j = 0
while pl:
    win.fill(BLACK)
    clock.tick(FPS)
    #обработка нажатий на кнопки
    # Отрисовка - в первую очередь!
    # Движение палок
    functions.update(st1, st2, ball)
    functions.on_keys(st1, st2)
    functions.drawing(win, st1, st2, ball)
    functions.stick_moving(st1, st2)
    ball.moving(st1, st2, border_w, real_screen_h)
    # Явный (реакционный) алгоритм движения
    #
    if player2 == 0:
        if (st2.top <= ball.bot -15 and ball.top <= st2.bot - 15 ) and not start :            # Если шар не в палке
            st2.dir = STOP
        else:
            if st2.center.x >= ball.center.x: st2.dir = UP             # Если ниже
            if st2.center.x <= ball.center.x: st2.dir = DOWN          # Если выше

    # Явный предикционный алгоритм движения
    #
    if ball.i == 0:
        st1.pos = st1.predict_pos(ball, BORDER, R_SC_H)
        st2.pos = st2.predict_pos(ball, BORDER, R_SC_H)
        ball.i = 1

    # Определяем вертикальные блин размеры окна (Done)
    if start:
        st1.dir = UP
        st2.dir = DOWN
        if st2.bot >= max_bot:
            if not start == 1:
                max_bot_pr = max_bot
            else: start = 2
            functions.stick_moving(st1, st2)
            max_bot = st2.bot
        if st1.top <= min_top:
            if not start == 1:
                min_top_pr = min_top
            else: start = 2
            functions.stick_moving(st1, st2)
            min_top = st1.top
        if min_top == min_top_pr and max_bot == max_bot_pr:
            start = 0
            real_screen_h = R_SC_H = - min_top + max_bot - BA_S - 2 * BORDER# Все это нужно для рассчета эффективного размера окна, боже дай мне сил
            print(min_top, max_bot, R_SC_H)
            ball.vel.y = Y
            ball.vel.x = X
    else:
        if player1 == 1: pass
        elif player2 == 2:
            if st1.center.x < st1.pos - 15: st1.dir = DOWN
            elif st1.center.x > st1.pos + 15: st1.dir = UP
            else: st1.dir = STOP
        else:
            vel = (st1.ball.vel.x ** 2 + st1.ball.vel.y ** 2 ) ** 1/2
            in_vect = np.array([st1.center.x / SC_H - 0.5, st1.ball.center.x / SC_H - 0.5, st1.ball.center.y / SC_W -
                                0.5, st1.ball.vel.x / vel , st1.ball.vel.y / vel])  # Не забываем нормировать вектор :/
            print("\n - in_vect -- ", in_vect, " \n")
            prediction = ai_pols.prediction(in_vect)
            if prediction[0] == 0:
                st1.dir = DOWN
            elif prediction[0] == 1:
                st1.dir = STOP
            else:
                st1.dir = UP

    functions.update(st1, st2, ball)
    if NN_on:
       st1.NN.recount()

    pygame.display.update()

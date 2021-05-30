import os

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import pygame
from Stick import Stick
import functions
from Ball import Ball
from backpropagation import *
import numpy as np
import ai_pols
from globals import *

# player: 1 - есть игрок, 0 - явный алгоритм, -1 - Нейросеть (Keras), 1 - нейросеть ручная
if True:
    player1 = 1
    player2 = -1

    i = 1

    st1_pos = 0
    st2_pos = 0

    min_top = SC_H
    max_bot = 0
    min_top_pr = SC_H + 1
    max_bot_pr = -1

# инициализация
pygame.init()
# ai_pols.ai_init()  # Инициализируем нейросеть и обучаем её

ball = Ball(SC_W // 2, SC_H // 2, BA_S, BA_SPEED)
st1 = Stick(SC_W * 1 / 60, BORDER, ball)
st2 = Stick(SC_W * 59 / 60 - ST_W, 40, ball)

n_outputs = 3
n_inputs = 5
n_hidden = 4
network = initialize_network(n_inputs, n_hidden, n_outputs)

dataset = np.array([np.random.sample(6) - 0.5 for _ in range(2000)])
for i in range(len(dataset)):
    if dataset[i][0] > dataset[i][4]:
        dataset[i][5] = 2
    elif dataset[i][0] == dataset[i][4]:
        dataset[i][5] = 1
    elif dataset[i][0] < dataset[i][4]:
        dataset[i][5] = 0

test = dataset[-100:]
dataset = dataset[:-100]

accuracy_by_epoch = train_network(network, dataset, test, 0.5, 20, n_outputs)
plt.plot(accuracy_by_epoch)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

x_train = []
y_train = []
x_test = []
y_test = []
x_val = []
y_val = []

dataset1 = np.array([np.random.sample(5) - 0.5 for _ in range(2000)])
for i in range(len(dataset1)):
    input = (dataset1[i][0],
             dataset1[i][1],
             dataset1[i][2],
             dataset1[i][3],
             dataset1[i][4])
    if dataset1[i][0] > dataset1[i][4]:
        y_train.append(2)
    elif dataset1[i][0] == dataset1[i][4]:
        y_train.append(1)
    elif dataset1[i][0] < dataset1[i][4]:
        y_train.append(0)
    x_train.append(input)
x_val = x_train[-100:]
x_test = x_train[-200:-100]
x_train = x_train[:1799]
y_val = y_train[-100:]
y_test = y_train[-200:-100]
y_train = y_train[:1799]

print('Первая модель')

inputs = keras.Input(shape=(5, ), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(3, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Укажем параметры обучения (оптимизатор, функция потерь, метрики)
model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
              # Минимизируемая функция потерь
              loss=keras.losses.SparseCategoricalCrossentropy(),
              # Список метрик для мониторинга
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Обучим модель разбив данные на "пакеты" размером "batch_size", и последовательно итерируя весь датасет заданное количество "эпох"
print('# Обучаем модель на тестовых данных')
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=10,
                    # Мы передаем валидационные данные для мониторинга потерь и метрик на этих данных в конце каждой эпохи
                    validation_data=(x_val, y_val))

# Возвращаемый объект "history" содержит записи значений потерь и метрик во время обучения
print(history.history)

# Оценим модель на тестовых данных, используя "evaluate"
print('\n# Оцениваем на тестовых данных')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)



# Отрисовываем график зависимости точности при обучении и тесте от эпохи
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# Отрисовываем график величины потерь от эпохи
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

win = pygame.display.set_mode((SC_W + NN_WIDTH, SC_H))
pygame.display.update()
clock = pygame.time.Clock()

# главный цикл
pl = True
st1.dir = STOP
st2.dir = STOP
start = 1  # Переменная для того, чтобы определить размер экрана (да-да, такой вот ерундой занимаемся)
j = 0
while pl:
    win.fill(BLACK)
    clock.tick(FPS)
    # обработка нажатий на кнопки
    # Отрисовка - в первую очередь!
    # Движение палок
    functions.update(st1, st2, ball)
    functions.on_keys(st1, st2)
    functions.stick_moving(st1, st2)
    functions.drawing(win, st1, st2, ball)
    ball.moving(st1, st2, border_w, real_screen_h)
    # Явный (реакционный) алгоритм движения
    #


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
            else:
                start = 2
            functions.stick_moving(st1, st2)
            max_bot = st2.bot
        if st1.top <= min_top:
            if not start == 1:
                min_top_pr = min_top
            else:
                start = 2
            functions.stick_moving(st1, st2)
            min_top = st1.top
        if min_top == min_top_pr and max_bot == max_bot_pr:
            start = 0
            real_screen_h = R_SC_H = - min_top + max_bot - BA_S - 2 * BORDER  # Все это нужно для рассчета эффективного размера окна, боже дай мне сил
            print(min_top, max_bot, R_SC_H)
            ball.vel.y = Y
            ball.vel.x = X
    else:
        if player1 == 1:
            # inputs = list()  # b_x_n, b_y_n, b_v_x_n, b_v_y_n, st_x_n
            vel = (ball.vel.x ** 2 + ball.vel.y ** 2) ** 1 / 2
            inputs = [ball.center.x / SC_H - 0.5, ball.center.y / SC_H - 0.5, ball.vel.x / vel, ball.vel.y / vel,
                      st1.center.x / SC_H - 0.5]
            pred = predict(network, inputs)
            if pred == 2: st1.dir = DOWN
            elif pred == 0: st1.dir = UP
            else: st1.dir = STOP
            # pass
        elif player1 == 2:
            if st1.center.x < st1.pos - 15:
                st1.dir = DOWN
            elif st1.center.x > st1.pos + 15:
                st1.dir = UP
            else:
                st1.dir = STOP
        else:
            vel = (st1.ball.vel.x ** 2 + st1.ball.vel.y ** 2) ** 1 / 2
            in_vect = np.array([st1.center.x / SC_H - 0.5, st1.ball.center.x / SC_H - 0.5, st1.ball.center.y / SC_W -
                                0.5, st1.ball.vel.x / vel, st1.ball.vel.y / vel])  # Не забываем нормировать вектор :/
            print("\n - in_vect -- ", in_vect, " \n")
            prediction = ai_pols.prediction(in_vect)
            if prediction[0] == 0:
                st1.dir = DOWN
            elif prediction[0] == 1:
                st1.dir = STOP
            else:
                st1.dir = UP

        if player2 == -1:
            vel = (ball.vel.x ** 2 + ball.vel.y ** 2) ** 1 / 2
            inputs = np.array(
                [ball.center.x / SC_H - 0.5, ball.center.y / SC_H - 0.5, ball.vel.x / vel, ball.vel.y / vel,
                 st2.center.x / SC_H - 0.5]).reshape((1, 5))
            x = []
            x.append(inputs)
            pred = model.predict(x)
            pred = np.argmax(pred)
            if pred == 2: st2.dir = DOWN
            elif pred == 0: st2.dir = UP
            else: st2.dir = STOP

        elif player2 == 0:
            if (st2.top <= ball.bot - 15 and ball.top <= st2.bot - 15) and not start:  # Если шар не в палке
                st2.dir = STOP
            else:
                if st2.center.x >= ball.center.x: st2.dir = UP  # Если ниже
                if st2.center.x <= ball.center.x: st2.dir = DOWN  # Если выше
    functions.update(st1, st2, ball)
    if NN_on:
        st1.NN.recount()

    pygame.display.update()

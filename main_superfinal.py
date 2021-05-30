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

#player1: 1-ручная нейросеть с маленьким набором данных
#         2-ручная нейросеть с большим набором данных
#         3-реакционный алгоритм

#player2: 1-нейросеть keras с маленьким набором данных
#         2-нейросеть keras с большим набором данных
#         0-???

## player: 1 - есть игрок, 0 - явный алгоритм, -1 - Нейросеть (Keras), 1 - нейросеть ручная
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

ball = Ball(SC_W // 2, SC_H // 2, BA_S, BA_SPEED)
st1 = Stick(SC_W * 1 / 60, BORDER, ball)
st2 = Stick(SC_W * 59 / 60 - ST_W, 40, ball)

n_outputs = 3
n_inputs = 5
n_hidden = 4

# Инициализируем ручную нейросеть и обучаем её на маленьком наборе данных

network_small = initialize_network(n_inputs, n_hidden, n_outputs)

dataset_small = np.array([np.random.sample(6) - 0.5 for _ in range(100)])
for i in range(len(dataset_small)):
    if dataset_small[i][0] > dataset_small[i][4]:
        dataset_small[i][5] = 2
    elif dataset_small[i][0] == dataset_small[i][4]:
        dataset_small[i][5] = 1
    elif dataset_small[i][0] < dataset_small[i][4]:
        dataset_small[i][5] = 0

test_small = dataset_small[-10:]
dataset_small = dataset_small[:-10]

accuracy_by_epoch = train_network(network_small, dataset_small, test_small, 0.5, 5, n_outputs)
plt.plot(accuracy_by_epoch)
plt.title('Model accuracy small')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

# Инициализируем ручную нейросеть и обучаем её на большом наборе данных

network_big = initialize_network(n_inputs, n_hidden, n_outputs)

dataset_big = np.array([np.random.sample(6) - 0.5 for _ in range(2000)])
for i in range(len(dataset_big)):
    if dataset_big[i][0] > dataset_big[i][4]:
        dataset_big[i][5] = 2
    elif dataset_big[i][0] == dataset_big[i][4]:
        dataset_big[i][5] = 1
    elif dataset_big[i][0] < dataset_big[i][4]:
        dataset_big[i][5] = 0

test_big = dataset_big[-100:]
dataset_big = dataset_big[:-100]

accuracy_by_epoch = train_network(network_big, dataset_big, test_big, 0.5, 20, n_outputs)
plt.plot(accuracy_by_epoch)
plt.title('Model accuracy big')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Инициализируем нейросеть keras и обучаем её на маленьком наборе данных

x_train_small = []
y_train_small = []
x_test_small = []
y_test_small = []
x_val_small = []
y_val_small = []

dataset1_small = np.array([np.random.sample(5) - 0.5 for _ in range(100)])
for i in range(len(dataset1_small)):
    input = (dataset1_small[i][0],
             dataset1_small[i][1],
             dataset1_small[i][2],
             dataset1_small[i][3],
             dataset1_small[i][4])
    if dataset1_small[i][0] > dataset1_small[i][4]:
        y_train_small.append(2)
    elif dataset1_small[i][0] == dataset1_small[i][4]:
        y_train_small.append(1)
    elif dataset1_small[i][0] < dataset1_small[i][4]:
        y_train_small.append(0)
    x_train_small.append(input)
x_val_small = x_train_small[-10:]
x_test_small = x_train_small[-20:-10]
x_train_small = x_train_small[:79]
y_val_small = y_train_small[-10:]
y_test_small = y_train_small[-20:-10]
y_train_small = y_train_small[:79]

inputs = keras.Input(shape=(5, ), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(3, activation='softmax', name='predictions')(x)

model_small = keras.Model(inputs=inputs, outputs=outputs)

# Укажем параметры обучения (оптимизатор, функция потерь, метрики)
model_small.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
              # Минимизируемая функция потерь
              loss=keras.losses.SparseCategoricalCrossentropy(),
              # Список метрик для мониторинга
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Обучим модель разбив данные на "пакеты" размером "batch_size", и последовательно итерируя весь датасет заданное количество "эпох"
print('# Обучаем модель на тестовых данных')
history = model_small.fit(x_train_small, y_train_small,
                    batch_size=64,
                    epochs=5,
                    # Мы передаем валидационные данные для мониторинга потерь и метрик на этих данных в конце каждой эпохи
                    validation_data=(x_val_small, y_val_small))

# Возвращаемый объект "history" содержит записи значений потерь и метрик во время обучения
print(history.history)

# Оценим модель на тестовых данных, используя "evaluate"
print('\n# Оцениваем на тестовых данных')
results = model_small.evaluate(x_test_small, y_test_small, batch_size=128)
print('test loss, test acc:', results)



# Отрисовываем график зависимости точности при обучении и тесте от эпохи
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Model small accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Отрисовываем график величины потерь от эпохи
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model small loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Инициализируем нейросеть keras и обучаем её на большом наборе данных

x_train_big = []
y_train_big = []
x_test_big = []
y_test_big = []
x_val_big = []
y_val_big = []

dataset1_big = np.array([np.random.sample(5) - 0.5 for _ in range(2000)])
for i in range(len(dataset1_big)):
    input = (dataset1_big[i][0],
             dataset1_big[i][1],
             dataset1_big[i][2],
             dataset1_big[i][3],
             dataset1_big[i][4])
    if dataset1_big[i][0] > dataset1_big[i][4]:
        y_train_big.append(2)
    elif dataset1_big[i][0] == dataset1_big[i][4]:
        y_train_big.append(1)
    elif dataset1_big[i][0] < dataset1_big[i][4]:
        y_train_big.append(0)
    x_train_big.append(input)
x_val_big = x_train_big[-100:]
x_test_big = x_train_big[-200:-100]
x_train_big = x_train_big[:1799]
y_val_big = y_train_big[-100:]
y_test_big = y_train_big[-200:-100]
y_train_big = y_train_big[:1799]

inputs = keras.Input(shape=(5, ), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(3, activation='softmax', name='predictions')(x)

model_big = keras.Model(inputs=inputs, outputs=outputs)

# Укажем параметры обучения (оптимизатор, функция потерь, метрики)
model_big.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
              # Минимизируемая функция потерь
              loss=keras.losses.SparseCategoricalCrossentropy(),
              # Список метрик для мониторинга
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Обучим модель разбив данные на "пакеты" размером "batch_size", и последовательно итерируя весь датасет заданное количество "эпох"
print('# Обучаем модель на тестовых данных')
history = model_big.fit(x_train_big, y_train_big,
                    batch_size=64,
                    epochs=20,
                    # Мы передаем валидационные данные для мониторинга потерь и метрик на этих данных в конце каждой эпохи
                    validation_data=(x_val_big, y_val_big))

# Возвращаемый объект "history" содержит записи значений потерь и метрик во время обучения
print(history.history)

# Оценим модель на тестовых данных, используя "evaluate"
print('\n# Оцениваем на тестовых данных')
results = model_big.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)



# Отрисовываем график зависимости точности при обучении и тесте от эпохи
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Model big accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Отрисовываем график величины потерь от эпохи
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model big loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

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
        if player1 == 1: #ручная нейросеть с маленьким набором данных
            vel = (ball.vel.x ** 2 + ball.vel.y ** 2) ** 1 / 2
            inputs = [ball.center.x / SC_H - 0.5, ball.center.y / SC_H - 0.5, ball.vel.x / vel, ball.vel.y / vel,
                      st1.center.x / SC_H - 0.5]
            pred = predict(network_small, inputs)
            if pred == 2: st1.dir = DOWN
            elif pred == 0: st1.dir = UP
            else: st1.dir = STOP
        elif player1 == 3: #реакционный алгоритм
            if st1.center.x < st1.pos - 15:
                st1.dir = DOWN
            elif st1.center.x > st1.pos + 15:
                st1.dir = UP
            else:
                st1.dir = STOP
        if player1 == 2: #ручная нейросеть с большим набором данных
            vel = (ball.vel.x ** 2 + ball.vel.y ** 2) ** 1 / 2
            inputs = [ball.center.x / SC_H - 0.5, ball.center.y / SC_H - 0.5, ball.vel.x / vel, ball.vel.y / vel,
                      st1.center.x / SC_H - 0.5]
            pred = predict(network_big, inputs)
            if pred == 2: st1.dir = DOWN
            elif pred == 0: st1.dir = UP
            else: st1.dir = STOP
        '''
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
        '''

        if player2 == 1: #нейросеть keras с маленьким набором данных
            vel = (ball.vel.x ** 2 + ball.vel.y ** 2) ** 1 / 2
            inputs = np.array(
                [ball.center.x / SC_H - 0.5, ball.center.y / SC_H - 0.5, ball.vel.x / vel, ball.vel.y / vel,
                 st2.center.x / SC_H - 0.5]).reshape((1, 5))
            x = []
            x.append(inputs)
            pred = model_small.predict(x)
            pred = np.argmax(pred)
            if pred == 2: st2.dir = DOWN
            elif pred == 0: st2.dir = UP
            else: st2.dir = STOP
        elif player2 == 2: #нейросеть keras с большим набором данных
            vel = (ball.vel.x ** 2 + ball.vel.y ** 2) ** 1 / 2
            inputs = np.array(
                [ball.center.x / SC_H - 0.5, ball.center.y / SC_H - 0.5, ball.vel.x / vel, ball.vel.y / vel,
                 st2.center.x / SC_H - 0.5]).reshape((1, 5))
            x = []
            x.append(inputs)
            pred = model_big.predict(x)
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

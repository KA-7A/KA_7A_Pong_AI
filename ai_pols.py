import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
import numpy as np
from globals import *
from Stick import *
import random

EPOCHS = 10
BATCH = 100
model = tf.keras.Sequential()
def ai_init_and_learn(sticks_list):
    model.add(layers.Dense(5, activation='sigmoid', input_shape=(5,)))
    model.add(layers.Dense(5, activation='sigmoid', input_shape=(5,)))
    #model.add(layers.Dense(5, activation='relu'))
    #model.add(layers.Dense(5, activation='relu'))

    model.add(layers.Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    x_train = np.zeros((len(sticks_list), 5))
    for i in range(len(sticks_list)):
        speed = (((sticks_list[i].ball.vel.x)**2 + (sticks_list[i].ball.vel.x)**2)**0.5)
        x_train[i][0] = sticks_list[i].center.x / (SC_H - 2*BORDER)
        x_train[i][1] = sticks_list[i].ball.center.x / (SC_H - 2*BORDER)
        x_train[i][2] = sticks_list[i].ball.center.y / (SC_W - 2*BORDER)
        x_train[i][3] = sticks_list[i].ball.vel.x / speed
        x_train[i][4] = sticks_list[i].ball.vel.y / speed
        #print("speed", speed)
    # 2 - up, 1 - stop, 0 - down
    vect_train = np.zeros((len(sticks_list), 1))
    for i in range(len(sticks_list)):
        pos = sticks_list[i].predict_pos(sticks_list[i].ball, BORDER, SC_H - 2*BORDER - 10)
        if pos < sticks_list[i].top:
            vect_train[i] = 2
        elif pos > sticks_list[i].bot:
            vect_train[i] = 0
        else: vect_train[i] = 1

    y_train = keras.utils.to_categorical(vect_train, num_classes=3)

    if False:
        x_test    = np.random.random((3, 5))
        vect_test = np.random.randint(3, size=(3, 1))
        for i in range(3):
            test_ball  = Ball(x_test[i][2]*(SC_W - 2*BORDER)-5, x_test[i][1]*(SC_W - 2*BORDER), 10, (x_test[i][3], x_test[i][4]))
            test_stick = Stick(SC_W/60, x_test[0]*(SC_H - 2*BORDER), test_ball)
            pos = test_stick.predict_pos(test_ball, BORDER, SC_H - 2*BORDER - 10)
            if pos < sticks_list[i].top:
                vect_test[i] = 2
            elif pos > sticks_list[i].bot:
                vect_test[i] = 0
            else: vect_test[i] = 1

        y_test = keras.utils.to_categorical(vect_test , num_classes=3)
        print("x_test --  \n", x_test)
        print("y_test -- \n", y_test)

    print("x_train -- \n", x_train)
    print("y_train -- \n", y_train)
    model.fit(x_train, y_train, epochs=5, batch_size=BATCH)

    #score = model.evaluate(x_test, y_test, batch_size=10)   # Предсказывающая функция
    print("\n------------------------\n")
    #print(score)
    model.save_weights('./weights/my_model')

def prediction(in_vect):
    #print("\n------------------------\n")
    speed = (((in_vect[3])**2 + (in_vect[4])**2)**0.5)
    x_test = np.zeros((1, 5))
    x_test[0][0] = in_vect[0] / (SC_H - 2*BORDER)
    x_test[0][1] = in_vect[1] / (SC_H - 2*BORDER)
    x_test[0][2] = in_vect[2] / (SC_W - 2*BORDER)
    x_test[0][3] = in_vect[3] / speed
    x_test[0][4] = in_vect[4] / speed
    print("\n\n  -  Speed  - - -  ", speed)
    res = model.predict_classes(x_test, batch_size=BATCH)
    print("\n -- res --", res)
    return res
    #print(prediction)



def ai_init():
    sticks_list = []
    balls_list  = []
    for i in range(100000):
        x_test = np.random.random((5,)) # Вектор на 5 рандомных значения

        balls_list. append(Ball(x_test[2]*(SC_W - 2*BORDER)-5, x_test[1]*(SC_W - 2*BORDER), 10, (x_test[3]*10, x_test[4]*10)))
        sticks_list.append(Stick(SC_W * 1 / 60,  random.randint(0, SC_H), balls_list[i]))
    ai_init_and_learn(sticks_list) # Запускаем нашу дискотеку (обучательную)

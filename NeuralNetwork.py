import numpy as np
import numpy.random as rand
import random
import math
import pygame

#зададим параметры

layers = 3      #количество слоев
n_out  = 1      #длина выходного вектора
#better_weights_main = np.zeros((n,layers))      #выбранные генетикой весы основные
#better_weights_out  = np.zeros((layers,n_out))  #выбранные генетикой весы выходного слоя
eps = 0.01                                      #в каких пределах будем искать веса

BLACK = (0  , 0  , 0  )
WHITE = (255, 255, 255)
BLUE  = (0  , 0  , 200)
RED   = (200, 0  , 0  )
GREEN = (0  , 200, 0  )
YELLOW= (255, 255, 0  )

#класс нейронки
#нужно еще подумать как получать лучшие веса с генетики (мб ее прописать в классе?)

class NeuralNetwork:
    def __init__(self,x):  
        print(x, len(x), "\n")
        self.weights_list = []                                              #Лист матриц переходов от одного слоя к другому
        self.layer_list = []    
        self.layer_list.append(np.asarray(x))                                 # Добавляем инпут слой
        if layers > 2:
            for i in range(layers-2):                                 
                self.layer_list.append(np.random.sample((len(x),1 )))                       # Закидываем  нулевые   слои подходящих размеров
        print("wat")
        self.layer_list.append(np.random.sample((n_out, 1)))   
        #self.layer_list.append(np.random.sample((n_out, 1)))# Закидываем  выходной  слой нужного  размера
        for i in range(layers-1):
            self.weights_list.append(np.random.sample((len(x), len(x)))-0.5)
        self.weights_list.append(np.random.sample((n_out, len(x)))-0.5)             # Mатрица для выходного слоя размером layers строк и n_out столбцов

        print("l_list", self.layer_list, "\n")
        print("len l_list", len(self.layer_list), "\n")
        print("w_list", self.weights_list, "\n")
        print("len w_list", len(self.weights_list), "\n")

        self.recount()                                          # Последний слой подтягивается до максимального размера тут.. Почему-то :/
        print("l_list", self.layer_list, "\n")
        print("len l_list", len(self.layer_list), "\n")
        print("w_list", self.weights_list, "\n")
        print("len w_list", len(self.weights_list), "\n")

        self.out = self.layer_list[layers-1]                    # Выходной вектор

    def recount(self):
        for i in range(layers-1):                               # Не выходя за границы массива пересчитываем значения на каждом слое
            self.layer_list[i+1]=np.dot(self.weights_list[i], self.layer_list[i])
        
        #print(self.layer_list)
        #print()
        #print(self.weights_list)

    def feedback(self): #изменение весов
        self.weights_main = better_weights_main + self.dweights #меняем веса
        self.weights_out  = better_weights_out  + self.dweights
        

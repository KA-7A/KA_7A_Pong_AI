# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp


# Загрузка CSV файла с данными
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Преобразовываем string колонку во float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Преобразовываем string колонку в integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Находим min и max значения для каждой колонки
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Нормализуем данные
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Разделяем данные на k частей
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Считаем метрику
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Оцениваем алгоритм с использование перекрестной проверки
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Считаем значение активации для нейрона
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Активация нейрона
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Распространение вперед от входного до выходного слоя
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            # inputs пустое
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Вычисляем производную выходного нейрона
def transfer_derivative(output):
    return output * (1.0 - output)


# Вычисляем ошибку обратного распространения и сохраняем в нейронах
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Обновляем веса с учетом ошибки
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Обучаем нейронную сеть на n эпохах
def train_network(network, train, test, l_rate, n_epoch, n_outputs):
    accuracy_by_epoch = []
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        predictions = list()
        for row in test:
            prediction = predict(network, row)
            predictions.append(prediction)
        actual = []
        for i in range(len(test)):
            actual.append(int(test[i][5]))
        accuracy_by_epoch.append(accuracy_metric(actual, predictions))
    return accuracy_by_epoch


# Инициализируем нейронную сеть
def initialize_network(n_inputs, n_hidden, n_outputs):  # везде стоят i
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in
                    range(n_outputs)] 
    network.append(output_layer)
    return network


# Предсказываем с помощью нейронной сети
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Алгоритм обратного распространения ошибки со стохастическим градиентным спуском
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    n_outputs = 3
    train_network(network, train, test, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        print(predict)
        predictions.append(prediction)
    return (predictions)

'''
if __name__ == "__main__":
    n_outputs = 3
    n_inputs = 6
    n_hidden = 4
    network = initialize_network(n_inputs, n_hidden, n_outputs)

    dataset = list()
    # [ input = (b_x_n, b_y_n, b_v_x_n, b_v_y_n, st_x_n, res) , ()
    # _n -- значения, нормированные на единицу, без этого суффикса -- обычные
    # [-0.5 , 0.5]    ->  [0, SC_W] = rez_data      -- перевод координат из нормированных в обычные координаты
    # +0.5 ) * SC_W
    # vel = (ball.vel.x**2 + ball.vel.y**2) ** (1/2)    -- перевод скорости из норм в обычн.
    # ball.vel.x + 0.5) * vel   ???
    # end_pos = predict_pos(b_x, b_y, b_v_x, b_v_y, st_x)
    # -> st_x + end_pos -> res(moving_direction) == input[-1]

    # train_network(network, dataset, 0.5, 20, n_outputs)
    inputs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # list()     # b_x_n, b_y_n, b_v_x_n, b_v_y_n, st_x_n
    # inputs.append([0.0,0.0,0.0,0.0,0.0])
    res = predict(network, inputs)  # -> res

    if False:
        filename = 'seeds_dataset.csv'
        dataset = load_csv(filename)
        for i in range(len(dataset[0]) - 1):
            str_column_to_float(dataset, i)
        # convert class column to integers
        str_column_to_int(dataset, len(dataset[0]) - 1)
        # normalize input variables
        minmax = dataset_minmax(dataset)
        normalize_dataset(dataset, minmax)
        # evaluate algorithm
        n_folds = 5
        l_rate = 0.3
        n_epoch = 500
        n_hidden = 5
        scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
'''

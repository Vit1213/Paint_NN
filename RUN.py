import sys, os
import argparse
import TrainNN, Test, Make_TrainSet


def createParser(): # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument('--begin_learn', help="Запуск алгоритма обучения", action='store_const',
                        const=True)
    parser.add_argument('-e', '--epochs', help="Количество эпох тренеровки нейросети (по умолчанию 1000)", default=1000, type=int)
    parser.add_argument('-b', '--batch_size', help="Размер партии для тренеровки (по умолчанию 3 для экономии ОЗУ)", default=3,
                        type=int)
    parser.add_argument('-mg', '--make_gray', help="Сделать черно-белые изображения из цветных", action='store_const', const=True)
    parser.add_argument('-in', '--input_data', help="Путь к входных данным для обработки", default=None, type=str)
    parser.add_argument('-lrn_dt', '--learn_data', help="Путь к тренеровачному набору", default=None, type=str)
    parser.add_argument('--no_pridict', help="После обучения не обрабатывать картинки", action='store_const',
                        const=False)

if __name__ == "__main__":
    args = createParser()
    namespace = args.parse_args(sys.argv[1:])

    epochs = namespace.Iter
    batch_size = namespace.batch_size
    data_path = namespace.input_data
    learn_path = namespace.learn_data

    if namespace.make_gray:
        Make_TrainSet.make_gray_img(data_path)
    if namespace.begin_learn:
        TrainNN.begin_learn(epochs, batch_size, learn_path, namespace.no_pridict)

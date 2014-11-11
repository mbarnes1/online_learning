__author__ = 'mbarnes1'
from online_logistic_regression import OnlineLogisticRegression
import numpy as np


def main():
    """
    Opens data log and runs online logistic regression
    """
    log_reg = OnlineLogisticRegression(alpha=0.001, number_classes=5, number_features=10)
    labels_true = list()
    ins = open('data/oakland_part3_am_rf.node_features')
    results = open('results.txt', 'w')
    next(ins)  # first header line
    next(ins)  # first header line
    next(ins)  # first header line
    for counter, line in enumerate(ins):
        print 'Sample:', counter
        data = [float(x) for x in line.split()]
        label = map_label(int(data[4]))
        features = np.array(data[5:])
        log_reg.update(features, label)
        labels_true.append(label)
    accuracy = float(sum(np.array(labels_true) == np.array(log_reg.predictions)))/len(labels_true)
    results.write('*** Online Results *** \nNet accuracy: ' + str(accuracy) + '\n')
    class_accuracies = class_accuracy(labels_true, log_reg.predictions)
    name_accuracies = {map_name(label): acc for label, acc in class_accuracies.iteritems()}
    for name, accuracy in name_accuracies.iteritems():
        results.write(name + ': ' + str(accuracy) + '\n')
    ins = open('data/oakland_part3_an_rf.node_features')
    next(ins)  # first header line
    next(ins)  # first header line
    next(ins)  # first header line
    heldout_true = list()
    heldout_predicted = list()
    for counter, line in enumerate(ins):
        print 'Sample:', counter
        data = [float(x) for x in line.split()]
        label = map_label(int(data[4]))
        features = np.array(data[5:])
        heldout_predicted.append(log_reg.predict(features))
        heldout_true.append(label)
    accuracy = float(sum(np.array(heldout_true) == np.array(heldout_predicted)))/len(heldout_true)
    results.write('\n*** Held-out Results *** \nNet accuracy: ' + str(accuracy) + '\n')
    class_accuracies = class_accuracy(heldout_true, heldout_predicted)
    name_accuracies = {map_name(label): acc for label, acc in class_accuracies.iteritems()}
    for name, accuracy in name_accuracies.iteritems():
        results.write(name + ': ' + str(accuracy) + '\n')


def class_accuracy(labels_true, labels_predicted):
    """
    Calcultes the accuracy of each class
    :param labels_true: Vector of class labels
    :param labels_predicted: Vector of predicted labels
    :return accuracies: List of accuracies for each class
    """
    labels_true = np.array(labels_true)
    labels_predicted = np.array(labels_predicted)
    classes = set()
    for label in labels_true:
        classes.add(label)
    accuracies = dict()
    for cls in classes:
        true_indices = labels_true == cls
        occurences = sum(true_indices)
        correct = sum(labels_predicted[true_indices] == cls)
        accuracies[cls] = float(correct)/occurences
    return accuracies


def map_label(label):
    """
    Maps labels to zero indexed, sequential values
    :param label: The unmapped label
    :return mapped: Mapped label
    """
    label_dict = {
        1004: 0,  # veg
        1100: 1,  # wire
        1103: 2,  # pole
        1200: 3,  # ground
        1400: 4,  # facade
    }
    return label_dict[label]


def map_name(label):
    """
    Maps [0,4] label to clas sname
    :param label: Label 0, 1, 2, 3, or 4
    :return name: String
    """
    name_dict = {
        0: 'Veg',
        1: 'Wire',
        2: 'Pole',
        3: 'Ground',
        4: 'Facade',
    }
    return name_dict[label]


if __name__ == '__main__':
    main()
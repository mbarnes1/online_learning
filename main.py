__author__ = 'mbarnes1'
from online_logistic_regression import OnlineLogisticRegression
import numpy as np
from itertools import izip
import random


def main():
    """
    Opens data log and runs online logistic regression
    """
    log_reg = OnlineLogisticRegression(alpha=.001, number_classes=5, number_features=10)
    ins = open('data/oakland_part3_am_rf.node_features')
    results = open('results.txt', 'w')
    next(ins)  # first header line
    next(ins)  # first header line
    next(ins)  # first header line
    features = list()
    labels_true = list()
    for counter, line in enumerate(ins):
        print 'Loading sample:', counter
        data = [float(x) for x in line.split()]
        labels_true.append(map_label(int(data[4])))
        features.append(np.array(data[5:]))
    features = normalize(features)
    shuffle(features, labels_true)
    #features_resampled, labels_resampled, xyz = resample(100000, features, labels_true, xyz)
    for counter, (feature, label) in enumerate(izip(features, labels_true)):
        print 'Training sample:', counter
        log_reg.update(feature, label)
    correct = sum(np.array(labels_true) == np.array(log_reg.predictions))
    accuracy = float(correct)/len(labels_true)
    results.write('*** Online Results *** \nNet accuracy: ' + str(correct) + '/' + str(len(labels_true)) + ' = ' + str(accuracy) + '\n')
    class_accuracies = class_accuracy(labels_true, log_reg.predictions)
    name_accuracies = {map_name(label): acc for label, acc in class_accuracies.iteritems()}
    for name, (correct, occurences, accuracy) in name_accuracies.iteritems():
        results.write(name + ': ' + str(correct) + '/' + str(occurences) + ' = ' + str(accuracy) + '\n')
    ins = open('data/oakland_part3_an_rf.node_features')
    next(ins)  # first header line
    next(ins)  # first header line
    next(ins)  # first header line
    heldout_true = list()
    heldout_predicted = list()
    xyz = list()
    features = list()
    for counter, line in enumerate(ins):
        print 'Loading test sample:', counter
        data = [float(x) for x in line.split()]
        xyz.append(data[0:3])
        label = map_label(int(data[4]))
        features.append(np.array(data[5:]))
        heldout_true.append(label)
    features = normalize(features)
    for counter, feature in enumerate(features):
        print 'Evaluating test sample:', counter
        heldout_predicted.append(log_reg.predict(feature))
    xyz = np.array(xyz)
    correct = sum(np.array(heldout_true) == np.array(heldout_predicted))
    accuracy = float(correct)/len(heldout_true)
    results.write('\n*** Held-out Results *** \nNet accuracy: ' + str(correct) + '/' + str(len(heldout_true)) + ' = ' + str(accuracy) + '\n')
    class_accuracies = class_accuracy(heldout_true, heldout_predicted)
    name_accuracies = {map_name(label): acc for label, acc in class_accuracies.iteritems()}
    for name, (correct, occurences, accuracy) in name_accuracies.iteritems():
        results.write(name + ': ' + str(correct) + '/' + str(occurences) + ' = ' + str(accuracy) + '\n')
    np.savetxt('xyz.csv', xyz, delimiter=',')
    labels_unmapped = [unmap_label(x) for x in heldout_predicted]
    np.savetxt("labels.csv", labels_unmapped, delimiter=",")


def normalize(features):
    """
    Normalizes features. Zero mean and divide by standard deviation
    :param features: List of features
    :return:
    """
    features_array = np.array(features)
    mean = np.mean(features_array, axis=0)
    features_array = features_array-mean
    stddev = np.std(features_array, axis=0)
    features_array = features_array/stddev
    return list(np.nan_to_num(features_array))


def resample(n_samples, features, labels, xyz):
    """
    Shuffles and balances the classes
    :param features: List of features
    :param labels: List of labels
    :param xyz: List of xyz coordinates
    :return:
    """
    all_labels = set()
    for label in labels:
        all_labels.add(label)
    probabilities = np.zeros(len(labels))
    labels_vector = np.array(labels)
    for label in all_labels:
        label_indices = labels_vector == label
        probabilities += label_indices*sum(label_indices)
    probabilities = 1/probabilities
    probabilities = probabilities/sum(probabilities)
    indices = range(0, len(labels))
    resampled = np.random.choice(indices, size=n_samples, replace=True, p=probabilities)
    new_features = list()
    new_labels = list()
    new_xyz = list()
    for idx in resampled:
        new_features.append(features[idx])
        new_labels.append(labels[idx])
        new_xyz.append(xyz[idx])
    return new_features, new_labels, xyz


def shuffle(list_a, list_b):
    """
    Shuffles two lists in place such that order is maintained
    :param list_a:
    :param list_b:
    """
    combined = zip(list_a, list_b)
    random.shuffle(combined)
    list_a[:], list_b[:] = zip(*combined)


def class_accuracy(labels_true, labels_predicted):
    """
    Calcultes the accuracy of each class
    :param labels_true: Vector of class labels
    :param labels_predicted: Vector of predicted labels
    :return accuracies: Dictionary of [class, (correct, occurences, accuracy)]
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
        accuracies[cls] = (correct, occurences, float(correct)/occurences)
    return accuracies


def unmap_label(label):
    """
    Maps zero indexed labels to original values
    :param label: The mapped label
    :return mapped: The original label
    """
    label_dict = {
        0: 1004,  # veg
        1: 1100,  # wire
        2: 1103,  # pole
        3: 1200,  # ground
        4: 1400,  # facade
    }
    return label_dict[label]


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
import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

"""
Shuyan Liu
CS50's Intro to AI
07/08/2020
"""

TEST_SIZE = 0.4
MONTH_LIST = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        evidence = []
        labels = []
        for row in reader:
            currentEvidence = []
            currentEvidence.append(int(row[0]))
            currentEvidence.append(float(row[1]))
            currentEvidence.append(int(row[2]))
            currentEvidence.append(float(row[3]))
            currentEvidence.append(int(row[4]))
            currentEvidence.append(float(row[5]))
            currentEvidence.append(float(row[6]))
            currentEvidence.append(float(row[7]))
            currentEvidence.append(float(row[8]))
            currentEvidence.append(float(row[9]))
            currentEvidence.append(MONTH_LIST.index((row[10])))
            currentEvidence.append(int(row[11]))
            currentEvidence.append(int(row[12]))
            currentEvidence.append(int(row[13]))
            currentEvidence.append(int(row[14]))
            currentEvidence.append(returning_vistor(row[15]))
            currentEvidence.append(str_to_int(row[16]))
            evidence.append(currentEvidence)
            labels.append(str_to_int(row[17]))
    return (evidence, labels)

def returning_vistor(string):
    """
    Converts the "returning vistor" data into an integer
    """
    if string == "Returning_Visitor":
        return 1
    else:
        return 0

def str_to_int(string):
    """
    Converts the "weekend" and "revenue" data (represented by a string) into an integer
    """
    if string == "FALSE":
        return 0
    else:
        return 1
    
    
def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model =  KNeighborsClassifier(1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positiveCount = 0
    negativeCount = 0
    sensitivity = 0
    specificity = 0
    for i in range(len(labels)):
        if labels[i] == 0:
            negativeCount += 1
            if predictions[i] == 0:
                specificity += 1
        else:
            positiveCount += 1
            if predictions[i] == 1:
                sensitivity += 1
    sensitivity /= positiveCount
    specificity /= negativeCount
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()

import csv

def extract_known_labels(filepath):
    with open(filepath) as csvFile:
        reader = csv.reader(csvFile, delimiter = '\t', quotechar = '"')
        known_labels = []
        for row in reader:
                known_labels.append(row[0])
        known_labels = list(map(int, known_labels))
        return known_labels

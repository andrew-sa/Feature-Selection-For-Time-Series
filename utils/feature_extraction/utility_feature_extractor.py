import csv
import pandas as pd
import numpy as np

# Questa funzione vi permette di adattare il tsv all'input
# richiesto dalle funzioni extract_feature
# Se qualcosa non vi è chiara, basta lanciarla e vedere l'output


#Prende in input il path del file che volete convertire nel formato preposto
def adaptTimeSeries(dataset):
    with open('../Datasets/{0}/{0}_TRAIN.tsv'.format(dataset), 'r') as csvTrain, open('../Datasets/{0}/{0}_TEST.tsv'.format(dataset), 'r') as csvTest:
        reader_train = csv.reader(csvTrain)
        reader_test = csv.reader(csvTest)
        id = 0
        listOfValue = []
        listOfId = []
        listOfTime = []
        listOfClass = []
        listGeneric = []

        for row in reader_train:
            splitted = row[0].split('\t')
            listOfClass.append(splitted[0])
            for i in range(1,len(splitted)):
                listOfValue.append(float(splitted[i]))
                listOfTime.append(i)
                listOfId.append(id)
                listGeneric.append((id,i,(float(splitted[i]))))
            id += 1
        
        for row in reader_test:
            splitted = row[0].split('\t')
            listOfClass.append(splitted[0])
            for i in range(1,len(splitted)):
                listOfValue.append(float(splitted[i]))
                listOfTime.append(i)
                listOfId.append(id)
                listGeneric.append((id,i,(float(splitted[i]))))
            id += 1
        
        df = pd.DataFrame(listGeneric, columns=['id', 'time','value'])
        series = pd.Series((i for i in listOfClass))
        return df,series

'''
# Questa funzione vi permette di unire i file di testing e training nel caso ne aveste bisogno
# Basterà passare solamente il nome del dataset che vi interessa, tipo ECG5000, ECG200...
def mergeTSV(fileName):
    f = open(fileName+".tsv", "w+")
    index =["TRAIN","TEST"]
    for value in index:
        # Ovviamente cambiate il path
        with open("C:/Users/Donato/Desktop/UCRArchive_2018/"+fileName+"/"+fileName+"_"+ value+".tsv", 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                for i in range(0,len(row)):
                    f.write(row[0])
                f.write("\n")
    f.close()
'''

import csv

import DatasetConstants as dconst
import xlsxwriter


class Dataset:
    def __init__(self, datasetName):
        self.datasetName = datasetName

    def readDataset(self, firstN):
        with open(self.datasetName, 'r') as file:
            csvreader = csv.reader(file)
            dsResultsList = []
            for row in csvreader:
                dsRow = str(row[0]).split("\t")
                mainFolder = dsRow[0].rsplit("_", 1)[0]
                dsResultsList.append(DatasetRow(dsRow[5],
                                                dconst.videosDatasetPath + "\\" + mainFolder + "\\" + dsRow[0].replace(
                                                    "\"", "").replace(
                                                    "\'",
                                                    "") + ".avi", mainFolder))
        if firstN <= -1:
            return dsResultsList
        return dsResultsList[:firstN]

class DatasetRow:
    def __init__(self, description, videoPath, videoName):
        self.description = description
        self.videoPath = videoPath
        self.videoName = videoName

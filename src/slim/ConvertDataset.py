import re
import os
import csv
import numpy as np

dataPattern = re.compile(r"^\d+[:]$")

def convert(filepath):
    with open(filepath, mode="r") as infile:
        r = csv.reader(infile, delimiter=" ")
        for row in r:
            if dataPattern.match(row[0]):
                arr = np.zeros(97)
                for i in range(1, len(row)):
                    arr[int(row[i])] = 1
                    print (",".join(str(x) for x in arr) )

if __name__ == '__main__':
    convert("/home/jdyer1/Desktop/code_SDM2017/work/data/adult.db")
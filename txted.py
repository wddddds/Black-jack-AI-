import pandas as pd


with open('1.txt', 'r') as f:
    data = f.readlines()

f = open('2.txt', 'a')
for i in xrange(len(data)):
    f.write(data[i])
    f.write('\n')
import csv


def show(matrix):
    for line in matrix:
        print(line)


adj_matrix = []
cr = csv.reader(open("adjacenceMatrix.csv", "r"))
for i, val in enumerate(cr):
    adj_matrix.append(val)
show(adj_matrix)




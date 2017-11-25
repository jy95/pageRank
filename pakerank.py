import csv
import numpy as np


def pageRankScore(A: np.matrix, alpha: float = 0.9):
    # without astype : numpy thinks it is a matrix of string
    adj_matrix = A.astype(np.int)
    print("Starting the program : Matrix of shape %s with alpha %f " % (A.shape, alpha))
    print(adj_matrix)
    # Vector of the sum for each column
    indegree = adj_matrix.sum(axis=0).getA1()
    print("indegree of each Node")
    print(indegree)
    print("Computing the probability matrix")

    # help us to not call sum multiple time when we will modify the matrix
    outdegree = adj_matrix.sum(axis=1).getA1()

    probability_matrix = []
    counter = 0
    for line in adj_matrix:
        row = line.getA1() / outdegree[counter]
        probability_matrix.append(row)
        counter += 1

    probability_matrix = np.matrix(probability_matrix, np.float)
    print(probability_matrix)

    print("Computing the transition-probability matrix Pt")
    transition_probability_matrix = probability_matrix.transpose()
    print(transition_probability_matrix)


# Read the matrix from csv and transform it to numpy matrix
matrix = []
cr = csv.reader(open("adjacenceMatrix.csv", "r"))
for i, val in enumerate(cr):
    matrix.append(val)

adj_matrix_np = np.matrix(matrix)
pageRankScore(A=adj_matrix_np)

# Call with a custom alpha
# pageRankScore(A=adj_matrix_np,alpha=0.8)

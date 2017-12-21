import csv
import numpy as np
import fractions
# A more user-friendly way to print matrix as fractions
# credits to https://stackoverflow.com/a/42209716/6149867
np.set_printoptions(formatter={'all': lambda x: str(fractions.Fraction(x).limit_denominator())})


def pageRankScore(A: np.matrix, alpha: float = 0.9):
    # without astype : numpy thinks it is a matrix of string
    adj_matrix = A.astype(np.int)
    print("Starting the program : Matrix of shape %s with alpha %f \n %s" % (A.shape, alpha, adj_matrix))
    # Vector of the sum for each column
    in_degree = adj_matrix.sum(axis=0)
    print("indegree of each node \n %s" % in_degree)

    # help us to not call sum multiple time when we will modify the matrix
    out_degree = adj_matrix.sum(axis=1).getA1()

    probability_matrix = []
    counter = 0
    for line in adj_matrix:
        row = line.getA1() / out_degree[counter]
        probability_matrix.append(row)
        counter += 1

    probability_matrix = np.matrix(probability_matrix, np.float)
    print("Computing the transition-probability matrix \n %s" % probability_matrix)

    print("Init vector (using in_degree and normalize it);")
    vector = in_degree.transpose()
    # Now time to normalize this vector by the sum
    vector = vector / vector.sum()
    print(vector)

    # Relative error
    epsilon = pow(10, -8)
    print("Power method iteration (left eigenvector) of the google matrix with an error of %s" % epsilon)
    # Number of nodes (number of columns inside the probability matrix)
    n = probability_matrix.shape[1]  # tuple shape : rows, columns
    # vector
    vector_google = vector.transpose()
    # column vector : Full of ones line vector
    et = np.ones(n)
    # Vector's norms
    norm = np.linalg.norm(vector_google, ord=1)
    new_norm = 0
    # google matrix
    print("Google matrix : ")
    google = (alpha*probability_matrix)+((1-alpha)/n)*et
    print(google)
    print("Iterations now begins : ")
    # counter for iteration
    step = 1
    while abs(new_norm-norm) / norm > epsilon:
        print("\t Iteration n° %s" % step)
        norm = np.linalg.norm(vector_google, ord=1)
        vector_google = vector_google * google
        new_norm = np.linalg.norm(vector_google, ord=1)
        """ Just a way to print only the first 3 iterations """
        if step in [1, 2, 3]:
            print("\t\t Computed PageRank score : \n \t\t %s " % vector_google)
        step = step + 1
    print("The final PageRank score is : \n \t %s " % vector_google)


# Read the matrix from csv and transform it to numpy matrix
def main():

    matrix = []
    cr = csv.reader(open("adjacenceMatrix.csv", "r"))
    for i, val in enumerate(cr):
        matrix.append(val)

    adj_matrix_np = np.matrix(matrix)
    pageRankScore(A=adj_matrix_np)


if __name__ == "__main__": main()

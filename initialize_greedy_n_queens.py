import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

# def total_number_diagonal_conflicts(column : int, row : int, greedy_init : list):
#     diagonal_conflicts = 0

#     for i in range(column):
#         if np.abs(row - greedy_init[i]) == column - i:
#             diagonal_conflicts += 1

#     return diagonal_conflicts

# def another_version_total_number_diagonal_conflicts(column : int, row : int, state : list):
#     N = len(state)
#     conflicts = 0
#     for c, r in enumerate(state):
#         if r == row:  # same row
#             continue
#         if abs(c - column) == abs(r - row):  # diagonal
#             conflicts += 1
#     return conflicts

# def total_number_of_row_conflicts(greedy_init : list, column : int, row : int):
#     return np.count_nonzero(greedy_init[:column] == row)

# def select_best_random_candidate(conflicts):
#     candidates = np.where(conflicts == np.min(conflicts))[0]
#     return np.random.choice(candidates)


# def initialize_greedy_n_queens(N: int) -> list:
#     """
#     This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
#     assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
#     produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
#     algorithm's performance!

#     Example:
#     Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

#      _ _ _ _
#     |Q|_|_|_|
#     |_|_|Q|_|
#     |_|_|_|Q|
#     |_|Q|_|_|

#     which has one diagonal conflict between its two rightmost columns.

#     You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
#     as any additional import statements will be removed by the autograder.

#     :param N: integer representing the size of the NxN chessboard
#     :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
#     conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
#     """
#     greedy_init = np.zeros(N, dtype=int)
#     greedy_init[0] = np.random.randint(0, N)

#     ### YOUR CODE GOES HERE
#     for column in range(1,N):
#         conflicts = np.zeros(N, dtype=int)
#         for row in range(len(conflicts)):
#             conflicts[row] = total_number_of_row_conflicts(greedy_init, column, row) + total_number_diagonal_conflicts(column, row, greedy_init)
#         greedy_init[column] = select_best_random_candidate(conflicts)

#     return greedy_init


def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """
    greedy_init = np.zeros(N, dtype=int)
    diagr_conflicts = [0] * ((2 * N) - 1)
    diagl_conflicts = [0] * ((2 * N) - 1)
    row_conflicts = [0] * N

    set_of_rows = set(range(0,N))
    not_assigned = list()
    
    for col in range(N):
        row = set_of_rows.pop()

        conflicts = row_conflicts[row] +  diagr_conflicts[col + row] + diagl_conflicts[col + (N - row - 1)]
        if conflicts == 0:
            greedy_init[col] = row
            row_conflicts[row] += 1
            diagr_conflicts[col + row] += 1
            diagl_conflicts[col + (N - row - 1)] += 1
        else:
            set_of_rows.add(row)
            row2 = set_of_rows.pop()

            conflicts2 = row_conflicts[row2] +  diagr_conflicts[col + row2] + diagl_conflicts[col + (N - row2 - 1)]
            if conflicts2 == 0:
                greedy_init[col] = row2
                row_conflicts[row2] += 1
                diagr_conflicts[col + row2] += 1
                diagl_conflicts[col + (N - row2 - 1)] += 1
            else:
                set_of_rows.add(row2)
                not_assigned.append(col)
            

    
    for col in not_assigned:
        row = set_of_rows.pop()
        greedy_init[col] = row
        greedy_init[col] = row
        row_conflicts[row] += 1
        diagr_conflicts[col + row] += 1
        diagl_conflicts[col + (N - row - 1)] += 1
    

    return greedy_init



        

        


if __name__ == '__main__':
    for i in range(10):
        print(initialize_greedy_n_queens(4))
    # You can test your code here

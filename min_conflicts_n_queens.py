import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

def reassign(N):
    greedy_init = np.zeros(N, dtype=int)
    diagr_conflicts = [0] * ((2 * N) - 1)
    diagl_conflicts = [0] * ((2 * N) - 1)
    row_conflicts = [0] * N

    list_of_rows = list(x for x in range(0,N))
    not_assigned = list()
    
    for col in range(N):
        row = list_of_rows.pop(0)

        conflicts = row_conflicts[row] +  diagr_conflicts[col + row] + diagl_conflicts[col + (N - row - 1)]
        if conflicts == 0:
            greedy_init[col] = row
            row_conflicts[row] += 1
            diagr_conflicts[col + row] += 1
            diagl_conflicts[col + (N - row - 1)] += 1
        else:
            list_of_rows.append(row)
            row = list_of_rows.pop(0)

            conflicts = row_conflicts[row] +  diagr_conflicts[col + row] + diagl_conflicts[col + (N - row - 1)]
            if conflicts == 0:
                greedy_init[col] = row
                row_conflicts[row] += 1
                diagr_conflicts[col + row] += 1
                diagl_conflicts[col + (N - row - 1)] += 1
            else:
                list_of_rows.append(row)
                not_assigned.append(col)
            

    
    for col in not_assigned:
        row = list_of_rows.pop(0)
        greedy_init[col] = row
        greedy_init[col] = row
        row_conflicts[row] += 1
        diagr_conflicts[col + row] += 1
        diagl_conflicts[col + (N - row - 1)] += 1
    

    return greedy_init, row_conflicts, diagr_conflicts, diagl_conflicts

def max_conflict_column(current, row_conflicts, diagr_conflicts, diagl_conflicts):
    conflict = 0
    max_conflict = 0
    candidates = list()

    for col in range(len(current)):
        row = current[col]
        conflict = row_conflicts[row] + diagr_conflicts[col+row] + diagl_conflicts[col+(len(current)-row-1)]
        if conflict > max_conflict:
            max_conflict = conflict
            candidates = [col]
        
        elif conflict == max_conflict:
            candidates.append(col)
        
    return np.random.choice(candidates), max_conflict

def min_conflict_row(current, row_conflicts, diagr_conflicts, diagl_conflicts, col):
    conflict = 0
    min_conflict = len(current)
    candidates = []

    for row in range(len(current)):
        conflict = row_conflicts[row] + diagr_conflicts[col+row] + diagl_conflicts[col+(len(current)-row-1)]
        if conflict == 0:
            return row
        elif conflict < min_conflict:
            min_conflict = conflict
            candidates = [row]
        elif conflict == min_conflict:
            candidates.append(row)
    
    return np.random.choice(candidates)






def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """

    N = len(initialization)

    current, row_conflicts, diagr_conflicts, diagl_conflicts = reassign(N) # I need the conflict values
                                                                           # Note that technically initalization == current even after calling reassign(N)
    max_steps = 10000

    for idx in range(max_steps):
        result = max_conflict_column(current, row_conflicts, diagr_conflicts, diagl_conflicts)
        if result[1] > 3:
            best_row = min_conflict_row(current, row_conflicts, diagr_conflicts, diagl_conflicts, result[0])
            if best_row != current[result[0]]:
                col = result[0]
                row = current[result[0]]
                row_conflicts[row] += -1
                diagr_conflicts[col + row] += -1
                diagl_conflicts[col + (N- row - 1)] += -1

                current[result[0]] = best_row
                row_conflicts[best_row] += +1
                diagr_conflicts[col + best_row] += +1
                diagl_conflicts[col + (N - best_row - 1)] += +1

        elif result[1] == 3:
            return current, (idx+1)

    return [], max_steps

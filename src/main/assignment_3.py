import numpy as np

def function(t: float, y: float):
    return t - (y**2)

def eulers():
    original_y = 1.0
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10
    # set up h
    h = (end_of_t - start_of_t) / num_of_iterations
    for cur_iteration in range(0, num_of_iterations+1):
        # do we have all values ready?
        t = start_of_t
        y = original_y
        # create a function for the inner work
        inner_math = function(t, y)
        # this gets the next approximation
        next_y = y + (h * inner_math)
        # we need to set the just solved "y" to be the original y
        # and not only that, we need to change t as well
        start_of_t = t + h
        original_y = next_y
    return original_y

def range_kutta():
    y_start = 1.0
    t_start, t_end = (0, 2)
    num_of_iterations = 10
    h = (t_end-t_start) / num_of_iterations
    for i in range(0, num_of_iterations):
        t = t_start
        y = y_start

        k1 = h * function(t, y)
        k2 = h * function(t+(h/2), y+(k1/2))
        k3 = h * function(t+(h/2), y+(k2/2))
        k4 = h * function(t+h, y+k3)

        y_next = y + (1/6)*(k1+(2*k2)+(2*k3)+k4)

        t_start = t+h
        y_start = y_next
    return y_start

def gauss(A, b):
    n = len(b)
    # Combine A and b into augmented matrix
    Ab = np.concatenate((A, b.reshape(n,1)), axis=1)
    # print(Ab)
    # Perform elimination
    for i in range(n):
        # Find pivot row
        max_row = i
        for j in range(i+1, n):
            if abs(Ab[j,i]) > abs(Ab[max_row,i]):
                max_row = j

        # Swap rows to bring pivot element to diagonal
        Ab[[i,max_row], :] = Ab[[max_row,i], :] # operation 3 of row operations
        # Divide pivot row by pivot element
        pivot = Ab[i,i]
        Ab[i,:] = Ab[i,:] / pivot
        # Eliminate entries below pivot
        for j in range(i+1, n):
            factor = Ab[j,i]
            Ab[j,:] -= factor * Ab[i,:] # operation 2 of row operations
    # Perform back-substitution
    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            factor = Ab[j,i]
            Ab[j,:] -= factor * Ab[i,:]
    # Extract solution vector x
    x = Ab[:,n]
    return x

def calc_det(A):
    n = len(A[0])

    if n == 2:
        return A[0][0]*A[1][1] - A[0][1]*A[1][0]
    
    sum = 0.0
    for j in range(0, n):
        M = np.delete(A, 0, 0)
        M = np.delete(M, j, 1)
        calc = ((-1.0) ** j) * A[0,j] * calc_det(M)
        sum += calc
    
    return sum

def is_diag_dom(A):
    n = len(A)
    for i in range(0, n):
        sum = 0
        for j in range(0, n):
            if j == i:
                continue
            sum += abs(A[i][j])
        
        if abs(A[i][i]) < sum:
            return False
        
    return True

# def is_pos_def(A):



if __name__ == "__main__":
    ans1 = eulers()
    print("%.5f" % ans1)
    print()

    ans2 = range_kutta()
    print("%.5f" % ans2)
    print()

    A = np.array([[2,-1,1],
    [1,3,1],
    [-1,5,4]], dtype=np.double)
    b = np.array([6,0,-3], dtype=np.double)
    ans3 = gauss(A, b)
    print(ans3)
    print()

    A = np.array([[1, 1, 0, 3], 
                  [2, 1, -1, 1], 
                  [3, -1, -1, 2], 
                  [-1, 2, 3, -1]], dtype=np.double)
    
    ans4a = calc_det(A)
    print("%.5f" % ans4a)
    print()

    A = np.array([[9,0,5,2,1], 
                  [3,9,1,2,1], 
                  [0,1,7,2,3], 
                  [-4,2,3,12,2],
                  [3,2,4,0,8]], dtype=np.double)
    
    print(is_diag_dom(A))
    print()



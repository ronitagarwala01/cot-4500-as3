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

if __name__ == "__main__":
    ans1 = eulers()
    print("%.5f" % ans1)
    print()

    ans2 = range_kutta()
    print("%.5f" % ans2)
    print()

    

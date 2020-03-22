import numpy as np
def func1():
    house_size = [1, 2, 3]
    house_price = [2, 5, 6]
    noOfPoints = 3
    points=np.array([house_size,house_price], np.int32)
    start_c = 0
    start_m = 0
    print("[1] - Current Slope(M) : {0} and Y-Intercept(C) : {1} with (Error Value) : {2}".format(start_m,start_c,get_error(start_c, start_m, points,noOfPoints)))
    iterations_ctr = 10000
    learning_rate = 0.01
    [c, m] = get_M_C(points, start_c, start_m, learning_rate, iterations_ctr,noOfPoints)

def get_error(c, m, points,noOfPoints):
    errorVal = 0
    for i in range(0, noOfPoints):
        x = points[0, i]
        y = points[1, i]
        errorVal += (y - (m * x + c)) ** 2
    return errorVal / noOfPoints

def get_M_C(points, current_c, current_m, learning_rate, iterations_ctr,noOfPoints):
    c = current_c
    m = current_m
    for i in range(iterations_ctr):
        c, m = compute_C_M(c, m, points, learning_rate,noOfPoints)
        print("[{3}] - Current Slope(M) : {0} and Y-Intercept(C) : {1} with (Error Value) : {2}".format(m, c, get_error(c, m, points,noOfPoints),i+2))
    return [c, m]

def compute_C_M(c_current, m_current, points, learningRate,noOfPoints):
    c_gradient = 0
    m_gradient = 0
    N = noOfPoints
    for i in range(0, noOfPoints):
        x = points[0 , i]
        y = points[1 , i]
        c_gradient += -(2/N) * (y - ((m_current * x) + c_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + c_current))
    next_c = c_current - (learningRate * c_gradient)
    next_m = m_current - (learningRate * m_gradient)
    return [next_c, next_m]

func1()
### data taken for learning Gradient Descent
from numpy import *

def step_gradient_descent(b, m, data, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(data))
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        ### gradient is partial differentiation of error function which is average of square of cost function (actual_value-predicted_value)
        ## actual_value is obtained from dataset (y)
        ## predicted_value (since we using linear regression to fit a line) = (m * x) + c 
        ## cost = y-((m*x)+c)
        ## squared_error = (1/N)* cost**2
        b_gradient += (-2/N)*(y-((m*x)+b))
        m_gradient += (-2/N)*(y-((m*x)+b))*x
    
    ## updating the value of b & m with learning rate times the gradient values, so how long or short steps we take to update b & m depends heavily on learning rate, 
    ### if learning rate is too small, we may take very large time to converge
    ### if learning rate is too large, we may never converge.
    new_b = b - learning_rate * b_gradient
    new_m = m - learning_rate * m_gradient
    return new_b, new_m

def run_gradient_descent(data, learning_rate, num_of_iterations):
    ## initially the line(for linear regression) has slope = 0, intercept = 0
    ## we need to run gradient descent to get optimal value of b & m
    b = 0
    m = 0
    for i in range(num_of_iterations):
        ## get new value of b, m after multiple iterations of gradient descent and when we get optimal value the gradient becomes 0 and value of b & m are not updated further...
        b, m = step_gradient_descent(b, m, array(data), learning_rate)
    
    return [b, m]

def main():
    ## read data from csv
    data = genfromtxt("data.csv", delimiter=',')
    ## hyper-parameters
    learning_rate = 0.0001
    num_of_iterations = 1000
    [b, m] = run_gradient_descent(data, learning_rate, num_of_iterations)
    print("Optimum value of b = ", b)
    print("Optimum value of m = ", m)
    
if __name__ == "__main__":
    main()
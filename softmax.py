import numpy as np

softmax_output = [0.7, 0.1, 0.2]

softmax_output = np.array(softmax_output).reshape(-1, 1)

"""
Derivative of the softmax function's output equals
S.i,j * (delta.j,k - S.i,k) = S.i,j * delta.j,k - S.i,j * S.i,k

S.i,j denotes the j-th Softmax output of the i-th sample 

delta (Kronecker delta)
1: if i == j
0: if i != j
"""


# Kronecker delta can be calculated using a diagonal matrix
left_term = np.diagflat(softmax_output)

# for S.i,j * S.i,k multiplication of the softmax outputs can be done through a dot product
right_term = np.dot(softmax_output, softmax_output.T)

# final term
left_term - right_term

"""
resulting matrix is known as the Jacobian matrix 
"""

# WOW

import numpy as np
import neural_network as neurn

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)


X = X/np.amax(X, axis=0)
y = y/100


nn = neurn.Neural_Network()

xPredicted = np.array(([4,8]), dtype=float)
xPredicted = xPredicted / np.amax(xPredicted, axis=0)


# V2

for i in range(100000):
    print("Input: \n" + str(X))
    print("Actual Output: \n" + str(y))
    print("Predicted Output: \n" + str(nn.forward(X)))
    print("Loss: \n" + str(np.mean(np.square(y - nn.forward(X)))))
    print("\n")
    nn.train_AI(X, y)

nn.predict(xPredicted)



# V1

# o = nn.forward(X)
#
#
#
# print("Predicted output: \n" + str(o))
# print("Actual output: \n" + str(y))


from problem import MNISTProblemClass
import numpy as np
import matplotlib.pyplot as plt
p = MNISTProblemClass(classes =[0,1])
model = p.model0

#print model components
print(model)

# print number of parameters
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# print first 5 parameters of the first layer
print(list(model.parameters())[0][:1])

# list of 0 to 1 in 0.01 increments
x = np.arange(0.01, 1, 0.01)

y = 1/x *0.1

#plot the piecewise function
plt.plot(x, y)
plt.show()


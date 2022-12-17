from switching_optimizers.problem import NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem\
    ,RastriginProblem, SquareProblemClass, AckleyProblem, NormProblem
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from switching_optimizers.eval_functions import eval_agent, eval_handcrafted_optimizer, first_index_below_threshold
import torch 

# problem instance
problem = GaussianHillsProblem
#problem = NoisyHillsProblem()

xlim = 2
# plot the function
x = np.arange(-xlim, xlim, 0.1)
y = np.arange(-xlim, xlim, 0.1)
X, Y = np.meshgrid(x, y)
X, Y = torch.tensor(X), torch.tensor(Y)
Z = problem().function_def(X, Y)
Z = Z.detach().numpy()


plt.contourf(X, Y, Z)
plt.colorbar()
plt.show()

# plot the function in perspective
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
ax.set_axis_off()
plt.show()



#print(np.min(Z))  
#print(X)
#print(np.vstack([X.ravel(), Y.ravel()]))
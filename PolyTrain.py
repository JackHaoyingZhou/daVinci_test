
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from GaitAnaylsisToolkit.LearningTools.Trainer import TPGMMTrainer
from random import seed
from random import gauss
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

def coef(b, dt):

    A = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [1.0, dt, dt**2, dt**3],
                  [0, 1.0, 2*dt, 3*dt**2]])


    return np.linalg.pinv(A).dot(b)


# seed random number generator
seed(1)
count = 200
b = np.array([ [-0.267], [0.], [ -0.7 ], [0.0] ])
x = coef(b, 10)
fit = poly.Polynomial(x.flatten())
t = np.linspace(0,10,count)
y_prime = fit(t)
hip = []
for i in range(5):
    y = y_prime + gauss(-0.01, 0.01)
    hip.append(y)

b = np.array([ [0.221], [0.0], [0.5], [0.0] ])
x = coef(b, 10)
fit = poly.Polynomial(x.flatten())
t = np.linspace(0, 10, count)
y_prime = fit(t)
knee = []
for i in range(5):
    y = y_prime + gauss(-0.01, 0.01)
    knee.append(y)


b = np.array([ [-0.147], [0.0], [-0.2 ], [0.0] ])
x = coef(b, 10)
fit = poly.Polynomial(x.flatten())
t = np.linspace(0,10,count)
y_prime = fit(t)
ankle = []
for i in range(5):
    y = y_prime + gauss(-0.01, 0.01)
    ankle.append(y)


trainer = TPGMMTrainer.TPGMMTrainer(demo=[hip, knee, ankle, hip, knee, ankle],
                                    file_name="simpletest",
                                    n_rf=6,
                                    dt=0.01,
                                    reg=[1e-3],
                                    poly_degree=[3,3,3,3,3,3])
trainer.train()
runner = TPGMMRunner.TPGMMRunner("simpletest")


path = runner.run()

fig, axs = plt.subplots(3)


print(path)
for p in hip:
    axs[0].plot(p)
    axs[0].plot(path[:, 0], linewidth=4, color='black')

for p in knee:
    axs[1].plot(p)
    axs[1].plot(path[:, 1], linewidth=4, color='black')

for p in ankle:
    axs[2].plot(p)
    axs[2].plot(path[:, 2], linewidth=4, color='black')

plt.show()
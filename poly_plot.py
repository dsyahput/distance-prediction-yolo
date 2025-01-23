import numpy as np
import matplotlib.pyplot as plt

x = [1512, 979, 680, 511, 384, 308, 265, 211, 180, 156, 136, 119, 102, 93, 87, 81, 75]
y = [1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0, 3.25, 3.50, 3.75, 4.0, 4.25, 4.50, 4.75, 5.0]

mymodel = np.poly1d(np.polyfit(x, y, 4))
myline = np.linspace(min(x), max(x), 200)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))

plt.show()
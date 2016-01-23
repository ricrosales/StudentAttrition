__author__ = 'Ricardo'

from scipy import stats
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
rv = stats.beta(loc=.0001, scale=30, a=1.1, b=50)
# rv = stats.expon(loc=.00001, scale=10)
r = rv.rvs(size=1000000)
print(min(r), max(r))
ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2, bins=100)
plt.show()
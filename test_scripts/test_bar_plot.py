import numpy as np
import matplotlib.pyplot as plt

# —— Data from your sketch ——
# x-positions (you have bars at x = –2, –1, +1, +2)
x = np.array([-2, -1,  1,  2])

# Maximum values (diagonal hatch)
max_vals = np.array([ 0.01,  0.09,  0.002, -0.001])

# Minimum values (horizontal hatch)
min_vals = np.array([-0.001, -0.02,   0.1,   -0.002])

# bar width
w = 0.4

fig, ax = plt.subplots(figsize=(6,4))

# plot bars
ax.bar(x - w/2, max_vals, width=w,
       edgecolor='black', facecolor='none',
       hatch='\\\\', label='Maximum')
ax.bar(x + w/2, min_vals, width=w,
       edgecolor='black', facecolor='none',
       hatch='-', label='Minimum')

# draw axes through the origin
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

# ticks and labels
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_xlabel('x')
ax.set_ylabel('z')

# add arrow heads on the axes
# y-axis arrow
ylim = ax.get_ylim()
ax.annotate('', xy=(0, ylim[1]), xytext=(0, ylim[1]*0.9),
            arrowprops=dict(arrowstyle='->'))
# x-axis arrow
xlim = ax.get_xlim()
ax.annotate('', xy=(xlim[1], 0), xytext=(xlim[1]*0.9, 0),
            arrowprops=dict(arrowstyle='->'))

# legend
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

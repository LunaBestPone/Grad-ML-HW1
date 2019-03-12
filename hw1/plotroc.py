import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132,sharex=ax1,sharey=ax1)
ax3 = fig.add_subplot(133,sharex=ax1,sharey=ax1)
# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
ax1.plot([0.0, 0.030303030303030304, 0.06060606060606061, 0.21212121212121213, 0.36363636363636365, 1.0],[0.6851851851851852, 0.7592592592592593, 0.9629629629629629, 0.9814814814814815, 1.0, 1.0])
ax2.plot([0.0, 0.06060606060606061, 0.09090909090909091, 0.24242424242424243, 0.42424242424242425, 1.0],[0.7037037037037037, 0.9444444444444444, 0.9629629629629629, 0.9814814814814815, 1.0, 1.0])
ax3.plot([0.0, 0.030303030303030304, 0.06060606060606061, 0.09090909090909091, 0.30303030303030304, 0.42424242424242425, 1.0],[0.7037037037037037, 0.7407407407407407, 0.9444444444444444, 0.9629629629629629, 0.9814814814814815, 1.0, 1.0])
ax1.set_title("k = 20")
ax2.set_title("k = 25")
ax3.set_title("k = 30")

plt.show()

import matplotlib.pyplot as plt
import pickle
import sys

data = pickle.load(open(sys.argv[1], 'rb'))

fig, ax1 = plt.subplots()

x, y = zip(*[i for i in data['losses'].items() if i[1] >= 0])

ax1.semilogy(x, y, "g-")
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss', color='g')
ax1.tick_params('y', colors='g')
ax1.set_ylim(top=10)

x = data['test_sc'].keys()
y = [sum(i)/len(i) for i in data['test_sc'].values()]

ax2 = ax1.twinx()
ax2.plot(x, y, "y", linewidth=0.3)
ax2.set_ylabel('score', color='y')
ax2.tick_params('y', colors='y')

plt.show()

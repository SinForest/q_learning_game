import matplotlib.pyplot as plt
import pickle
import sys

data = pickle.load(open(sys.argv[1], 'rb'))

fig, ax1 = plt.subplots()

x, y = zip(*[i for i in data['losses'].items() if i[1] >= 0])

plt1 = ax1.plot(x, y, "g-", linewidth=0.7, label="train loss")
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss', color='g')
ax1.tick_params("y", colors='g')
ax1.set_ylim((0.1,1))

ax2 = ax1.twinx()
ax2.set_ylabel('score', color="darkorange")
ax2.tick_params("y", colors="darkorange")

x = data['test_sc'].keys()
y = [sum(i)/len(i) for i in data['test_sc'].values()]
plt2 = ax2.plot(x, y, "darkorange", linewidth=0.7, label="test score (avg)")

x = data['test_sc'].keys()
y = [max(i) for i in data['test_sc'].values()]
plt3 = ax2.plot(x, y, "darkorange", linewidth=0.3, label="test score (max)")

x, y = zip(*data['scores'].items())
plt4 = ax2.plot(x, y, "y", linewidth=0.1, label="train score")

plt.legend(handles=plt1+plt2+plt3+plt4, loc=2).get_frame().set_alpha(0.8)

plt.savefig("plot.png",bbox_inches='tight')
#plt.show()

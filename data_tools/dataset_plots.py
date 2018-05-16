from data_handler import *
import matplotlib.pyplot as plt

# dh = DataHandler('./data/EURUSD.csv')
# dh.timeSeriesToSupervised()
#
# data = dh.tsdata.values
# x, y = data[:, 1], data[:, 3]
# m = max(x)
#
# plt.plot(y)
# plt.ylim(0, m + 0.5)
# plt.xlim(0, 5050)
# plt.xlabel('Timestep (day)')
# plt.ylabel('Exchange Rate')
# plt.title('EUR\\USD Exchange Rate')
# plt.savefig('graphs/eurusd.eps', format='eps', dpi=1000)
# plt.show()

dh = DataHandler('./data/Sunspots.csv')
dh.timeSeriesToSupervised()

data = dh.tsdata.values
x, y = data[:, 1], data[:, 3]
m = max(x)

plt.plot(y)
plt.ylim(0, m + 15)
plt.xlim(0, 3225)
plt.xlabel('Timestep (month)')
plt.ylabel('Monthly Mean Total Sunspots')
plt.title('Monthly Mean Total Sunspots')
plt.savefig('graphs/sunspots.eps', format='eps', dpi=1000)
plt.show()

# dh = DataHandler('./data/mackey.csv')
# dh.timeSeriesToSupervised()
#
# data = dh.tsdata.values
# x, y = data[:, 1], data[:, 3]
# m = max(x)
#
# plt.plot(y)
# plt.ylim(0, m + 0.5)
# plt.xlim(0, 1000)
# plt.xlabel('Timestep')
# plt.ylabel('Equation Output')
# plt.title('Mackey-Glass Equation Output')
# plt.savefig('graphs/mackey.eps', format='eps', dpi=1000)
# plt.show()

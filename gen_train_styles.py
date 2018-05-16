"""
	Used to generate the graphs depicting the different train styles for ADLE.

	Author: Anthony Morast
	Date: 4/23/2018
"""

from data_handler import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random as rand

colors = ['red', 'blue', 'green', 'brown', 'black', 'olive', 'purple']
trainsize = 2000

dh = DataHandler('./data/EURUSD.csv')
dh.timeSeriesToSupervised()

data = dh.tsdata.values
x, y = data[:, 1], data[:, 3]
m = max(x)

plt.plot(y)
plt.ylim(0, m + 0.5)
plt.xlim(0, 5050)
plt.xlabel('Timestep (day)')
plt.ylabel('Price')
plt.title('EUR\\USD Exchange Rate (Sequential Training)')

# sequential
chunk_size = 400
for i in range(5):
    plt.gca().add_patch(patches.Rectangle(((i*chunk_size), 0),
                          chunk_size, m + 0.25, fill=False, ls='solid', ec=colors[i%len(colors)]))
    plt.text((i*chunk_size)+15, m+0.3, '$f_{' + str(i+1) + '}^*$')
plt.savefig('./paper/imgs/sequential_forex.eps')

# overlap
# base_size = 1000
# shift = int((trainsize-base_size) / 4)
# for i in range(5):
#     plt.gca().add_patch(patches.Rectangle(((i*shift), 0),
#                         base_size, m+0.25, fill=False, ls='solid',
#                         ec=colors[i % len(colors)]))
#     plt.text((i*shift)+15, m+0.3, '$f_{' + str(i+1) + '}^*$')
# plt.savefig('./paper/imgs/overlap_forex.eps')

# for i in range(10):
#     base = rand.randint(0, trainsize - 300)
#     size = rand.randint(300, 500)
#     while base + size > trainsize:
#         base = rand.randint(0, trainsize - 300)
#         size = rand.randint(300, 500)
#     plt.gca().add_patch(patches.Rectangle((base, 0), size, m+0.25, fill=False,
#                             ls='solid', ec=colors[i % len(colors)]))
# plt.savefig('./paper/imgs/random_segments_forex.eps')

plt.show()

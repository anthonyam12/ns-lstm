import matplotlib.pyplot as plt
from data_handler import *

dh = DataHandler("./data/mackey.csv")
dh.timeSeriesToSupervised()
dh.splitData(len(dh.tsdata) - 500, 500, 0)

train, test, _ = dh.getDataSets()
testy = test[:, 3]

pdh = DataHandler("./data_tools/arima_predictions_mackey.csv")
predictions = pdh.getColumnByHeader("x")
print(predictions)

plt.plot(testy, label='actual')
plt.plot(predictions, label='predicted')
plt.xlabel('Timestep')
plt.ylabel('Mackey-Glass Output')
plt.title('Actual vs. ARIMA Predicted Mackey-Glass Equation Test Set')
plt.legend()
# plt.savefig('./graphs/results/arima_results_mackey.eps', dpi=1000)
plt.show()

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

H = pd.DataFrame([[9663,142],[ 107,9698]])


cmap=plt.cm.Blues
classes = ["pos","neg"]
plt.imshow(H, interpolation='nearest', cmap=cmap)
plt.title("test")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.show()
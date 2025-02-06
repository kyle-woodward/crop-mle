from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)

# three 0s in pred, two in true so its true across top and pred down the side
# array([[2, 0, 0],
#        [0, 0, 1],
#        [1, 0, 2]])
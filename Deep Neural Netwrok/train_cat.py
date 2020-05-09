from dnn_app_utils import load_train_data, predict
from model import L_layer_model
import pickle
train_x_orig, train_y, classes = load_train_data()
train_x_flatten = train_x_orig.reshape(
    train_x_orig.shape[0], -1).T  # pylint: disable=unsubscriptable-object

train_x = train_x_flatten/255.
layers_dims = [12288, 20, 7, 5, 1]

print('Started Training, be patient please')
parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.0075,
                           num_iterations=3000, print_cost=True)

print('Training has finished')
filename = 'finalized_model.sav'
pickle.dump(parameters, open(filename, 'wb'))
predict(train_x, train_y, parameters, test=False)

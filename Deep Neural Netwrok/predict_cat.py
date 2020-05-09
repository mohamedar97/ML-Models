from dnn_app_utils import predict, load_test_data
import pickle
test_x_orig, test_y, classes = load_test_data()
test_x_flatten = test_x_orig.reshape(
    test_x_orig.shape[0], -1).T  # pylint: disable=unsubscriptable-object
test_x = test_x_flatten/255.
filename = 'finalized_model.sav'

parameters = pickle.load(open(filename, 'rb'))
predict(test_x, test_y, parameters, test=True)

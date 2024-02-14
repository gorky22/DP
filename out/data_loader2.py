import pickle
with open('Fruits.obj', 'wb') as fp:
    pickle.dump(banana, fp)


def get_train_data()
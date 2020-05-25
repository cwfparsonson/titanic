import pandas as pd
import tensorflow as tf
print('~*~*~* TensorFlow Version {} *~*~*~'.format(tf.__version__))
import numpy as np
import math



#def get_mini_batches(x, y, batch_size):
#    '''
#    Returns random mini batches from x, y data 
#    '''
#    random_idxs = np.random.choice(len(y), len(y), replace=False)
#    x_shuffled = x[random_idxs,:]
#    y_shuffled = y[random_idxs]
#
#    mini_batches = [(x_shuffled[i:i+batch_size,:], y_shuffled[i+i:batch_size]) for i in range(0,len(y),batch_size)]
#    
#    return mini_batches



if __name__ == '__main__':
    # load data
    df = pd.read_csv('files/processed_data.csv')

    # split into test and train
    x_train = df[pd.notnull(df['Survived'])].drop(['Survived'], axis=1)
    y_train = df[pd.notnull(df['Survived'])]['Survived']

    x_test = df[pd.isnull(df['Survived'])].drop(['Survived'], axis=1)

    # reshape labels to be tensors (to match dims of logits) and extract values
    # from pandas df
    y_train = y_train.iloc[:].values.reshape(-1,1)
    x_train = x_train.iloc[:].values
    #x_test = x_test.iloc[:].values

    print('x train:\n{}'.format(x_train))
    print('shape: {}'.format(x_train.shape))
    print('y train:\n{}'.format(y_train))
    print('shape: {}'.format(y_train.shape))

    
    # build nn model by stacking sequential layers
    # the first layer has the same number of neurons/units as there are dims
    # in the input data (1x9 = 9), and will flatten the data down to a 1D array
    # the second layer is a hidden layer with 20 nodes. TF will automatically
    # initialise weight values for each node, matrix multiply these weights by
    # the input data, and apply some bias to get the input to the activation func,
    # which will pass this processed data into the ReLU activation func
    # the final layer will take the output of this ReLU activation func again
    # by multiplying the outputs of ReLU func by some weights and adding biases
    # to get a vector output. This output is known as a 'logit', which is a vector 
    # of raw (non-normalised) predictions generated by a classification model. 
    # In this case, since we are using a binary classifier, we only need 1 neuron,
    # therefore the output logit of our nn will only have 1 element (the output of
    # our single neuron). The element(s) of the output logit can be considered
    # as the NN's non-normalised prediction that the datum that was input to
    # the NN was each class. Usually, these outputs are then normalised such that
    # summing all elements of the logit/prediction = 1 (using e.g. softmax func)
    # to give a proibability distribution of predictions across possible classes
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(units=20, activation='relu'),
        tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
        ])
    
    # as an example, use the first datum in x_train and the initialised NN model
    # to get a prediction/logit (non-normalised) from the NN
    print('~*~*~* EXAMPLE ~*~*~*')
    print('First datum input:\n{}'.format(x_train[:1]))
    predictions = model(x_train[:1]).numpy()
    print(x_train[:1].shape)
    print('Initialised model output logit:\n{}'.format(predictions))
    print('Shape of logit: {}'.format(predictions.shape))
    print('Shape of label: {}'.format(y_train[:1].shape))

    # can now convert logit elements into probabilities for each class using
    # softmax func to normalise
    #probs = tf.nn.softmax(predictions).numpy()
    probs = tf.nn.sigmoid(predictions).numpy()
    print('Probability dist over each class:\n{}'.format(probs))


    # can now define a loss func for update rule/optimiser/backprop to use. This
    # loss is the negative log probability of the true class: If returns 0, the model
    # is sure of the correct class. So far, we have not trained our model, therefore 
    # the untrained model will give random probabilies (1/2 for each class), therefore
    # the initial loss should be close to tf.log(1/2) ~= 0.69 
    #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # to get loss of prediction, pass logit predictions (not normalised) of NN
    # and the correct ground truth label into the loss function to eval the loss.
    # For our untrained NN, should be close to 0.69
    print('Label: {}'.format(y_train[:1]))
    loss = loss_fn(y_train[:1], predictions).numpy()
    print('Rough loss for untrained binary NN: {}'.format(-tf.math.log(0.5)))
    print('Actual loss by our untrained NN: {}'.format(loss))

    # can now train our NN model. Compile by setting the optimiser, loss func,
    # and the metrics to use 
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    # train model
    model.fit(x_train, y_train, epochs=10, batch_size=1)

    

    # use trained model to make predictions on test set
    test = pd.read_csv('files/test.csv')
    #prob_model = tf.keras.Sequential([model, tf.keras.activations.sigmoid()])
    #test['Survived'] = prob_model(x_test.iloc[:].values).numpy()
    test['Survived'] = model.predict(x_test.iloc[:].values)
    print(test['Survived'])
    test['Survived'] = test['Survived'].apply(lambda x: round(x, 0)).astype('int')
    print(test['Survived'])
    solution = test[['PassengerId', 'Survived']]
    
    # save solution
    solution.to_csv('files/nn_solution.csv', index=False)
    









    '''
    # 2 parts of a TensorFlow programme: (1) Definition (2) Prediction

    # PART 1: DEFINITION
    # initialise consts
    learning_rate = 0.5
    epochs = 100
    batch_size = 32

    # declare training data placeholders
    # input placeholder: (1x9) dims = 9
    #x = tf.placeholder(tf.float32, [None, x_train.shape[1]]) # None for any no. rows
    x = tf.Variable(tf.ones(shape=[None, x_train.shape[1]]), 
                    dtype=tf.float32,
                    name='x') # None=any no. rows
    # output placeholder: 1/0 output = 1 dim
    #y = tf.Variable(dtype=tf.float32, [None, 1])
    y = tf.Variable(tf.ones(shape=[None, 1]), 
                    dtype=tf.float32,
                    name='y')

    # setup i) weight & ii) bias vars for each layer
    # input hidden layer with 20 nodes in hidden layer & initialised weight values
    # following a random normal dist with mean 0 and standard dev 0.03:
    W1 = tf.Variable(tf.random_normal([x_train.shape[1],20], 
                     stddev=0.03),
                     name='W1')
    b1 = tf.Variable(tf.random_normal([20]), name='b1')
    # output hidden layer:
    W2 = tf.Variable(tf.random_normal([20, 1], 
                     stddev=0.03), 
                     name='W2')
    b2 = tf.Variable(tf.random_normal([1]), name='b2')

    # set up i) node input & ii) non-linear activation functions for hidden layer nodes
    # first, matrix multiply (matmul()) W1 by input vector x, and add() some bias b1.
    # This sets up the node input before passing to activation func:
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    # next, finalise hidden_out op by applying the chosen activation func (sigmoid)
    hidden_out = tf.math.sigmoid(hidden_out)

    # calc output of hidden layer (which is final output of our nn)
    # do this by again matrix multiplying output of hidden layer/act func 
    # by weights and then adding bias. Use softmax activation func for this
    # output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

    # define loss func for update rule/optimiser/backprop to use
    # first convert nn output y_ to a clipped output betwee 1e-10 and 0.999999
    # so that we never get log(0) (would return NaN and crash training)
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    # next, calc cross entropy (loss) of this output, which we want to minimise
    # reduce_sum takes sum of given axis of a tensor, and reduce_mean takes
    # the mean of a tensor
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1-y) * tf.log(1-y_clipped), axis=1))

    # set up optimiser to use simple grad descent. Specify learning rate for updates,
    # and specify that we want to minimise the cross_entropy func we've defined.
    # tensorflow will then perform gradient descent and backprop to optimise nn
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    
    # set up initialisation op
    init_op = tf.global_variables_initializer()

    # set up op for measuring accuracy of nn predictions
    # first set up op to determine if nn has made an accurate prediction. Use
    # equal() method to see if output of nn and label are equal, return tensor of booleans
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # next, calc mean accuracy from this tensor. Do this by first casting/converting
    # correc_prediction output from boolean tensor to float tensor, then use reduce_mean
    # to find mean of correct/wrong predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    mini_batches = get_mini_batches(x, y, batch_size)
    print(mini_batches)
    '''


    '''
    # PART 2: PREDICTION
    with tf.Session as sess:
        # initialise vars
        sess.run(init_op)
        total_batch = int(len(x_train)/batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = 
    '''
























    '''
    # def model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(17, input_dim=x_train.shape[1], activation='linear'))
    model.add(tf.keras.layers.Dense(8, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(8, activation='linear'))
    model.add(tf.keras.layers.Dense(8, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(8, activation='linear'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # train
    training = model.fit(x_train, 
                         y_train, 
                         epochs=100, 
                         batch_size=32, 
                         validation_split=0.2, 
                         verbose=0)
    val_acc = np.mean(training.history['val_accuracy'])
    print('\n%s: %.2f%%' % ('val_acc', val_acc*100))
    '''
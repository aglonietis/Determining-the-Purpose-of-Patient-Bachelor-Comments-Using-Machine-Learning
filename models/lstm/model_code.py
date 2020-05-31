# To use the model code, replace the getModel function in python_projects/notebooks/Training.ipynb notebook. It is the code of the model structrue

# create a model
def getModel(name):
    # get a model class
    model = keras.models.Sequential(name=name)

    # models layers
    model.add(layers.Embedding(vocabulary_size, vocab_vector_size, input_length=sentence_size, name='embed'))
    model.add(layers.SpatialDropout1D(0.25, name='spatial_dropout'))
    model.add(layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5, name='lstm_1'))
    model.add(layers.Dropout(0.5))

    # Last output layer
    model.add(layers.Dense(3, activation='softmax', name='dense_output'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

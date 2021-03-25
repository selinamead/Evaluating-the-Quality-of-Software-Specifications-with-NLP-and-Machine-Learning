from sklearn.base import BaseEstimator, TransformerMixin
from pandas.io.parsers import _get_col_names
import numpy as np
import pandas as pd


# function to seperate numerical and categorical columns of a input dataframe
def num_cat_separation(df):
    metrics_num = []
    metrics_cat = []
    for col in df.columns:
        if df[col].dtype == "object":
            metrics_cat.append(col)
        else:
            metrics_num.append(col)
    # print("Categorical columns : {}".format(metrics_cat))
    # print("Numerical columns : {}".format(metrics_num))
    return (metrics_cat, metrics_num)


# class to allow the selection of specific columns of a dataframe.
# It converts a dataframe selection into numpy array to be able to apply later scikitlearn functions on them
# https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, col_names):
        # print('\nImplementing DataFrameSelector...')
        self.col_names = col_names
        # print(col_names)

    def fit(self, X, y=None):
        # print('\n?Fitting DataFrameSelector')
        # print(X)
        return self

    def transform(self, X):
        # print('\n?Transforming DataFrameSelector')
        # print('Problem: indicies must be only integers, slices, ellipsis: Integer or boolean arrays are valid indices')
        # print('X values:\n', X)
        # print()
        # print('X.col_names.values:\n', X[self.col_names].values)
        return X[self.col_names].values


'''
def neural_networks():
			print('\nPreparing data for Neural Networks')

			num_classes = 5
			batch_size = 10 # The higher the batch size, the faster to train, but lower accuracy

			# remove req_proc

			# num_features = self.dfm_scaled.shape[1] 
			num_features = self.features_scaled.shape[1]
			# print('Num of Features = ', self.dfm_scaled.shape[1])
			print('Num of Features = ', self.features_scaled.shape[1])
			# feature_names = self.dfm_scaled.columns
			feature_names = self.features_scaled.columns
			print(feature_names)

			# for col in self.dfm_scaled.columns:
			# 	print(col)
			# for col in self.features_scaled.columns:
			# 	print(col)

    		# feature_columns = [col for col in dfm_scaled.columns]
    		# print(feature_columns) 

			# print('DFM SCALED: \n', self.dfm_scaled)
			# print('FEATURES_SCALED: \n', self.features_scaled)
			# input()

			# # get dataset
			scaled_features = self.features_scaled.copy() 
			# scaled_features['req_process'] = self.data['req_process']
			scaled_features['target'] = self.data['target']
			# scaled_data = self.features_scaled
			print('Scaled_Features df \n', scaled_features)

			# SPlit test and train data
			train_set, test_set = train_test_split(scaled_features, test_size=0.2, random_state=42) #0.1 test

			X_train = train_set.loc[:, train_set.columns != "target"]
			y_train = train_set.target

			X_test = test_set.loc[:, test_set.columns != "target"]
			y_test = test_set.target

			# Reshape training data 
			print('Shape of Training Data:', X_train.shape)
			X_train = np.array(X_train)

			# Reshape dimension of data to fit CNN model
			X_train_res = np.expand_dims(X_train, axis=2)
			X_test_res = np.expand_dims(X_test, axis=2)
			print('Reshaped Data for Neural Networks: ', X_train_res.shape)

			# encode train class values as integers
			encoder = LabelEncoder()
			encoder.fit(y_train)
			encoded_y = encoder.transform(y_train)
			# convert integers to dummy variables (i.e. one hot encoded)
			y_train = np_utils.to_categorical(encoded_y)

			# encode test class values as integers
			encoder = LabelEncoder()
			encoder.fit(y_test)
			encoded_y = encoder.transform(y_test)
			# convert integers to dummy variables (i.e. one hot encoded)
			y_test = np_utils.to_categorical(encoded_y)

			# # The maximum number of words to be used. (most frequent)
			# MAX_NB_WORDS = 50000
			# # Max number of words in each complaint.
			# MAX_SEQUENCE_LENGTH = 250
			# # This is fixed.
			# EMBEDDING_DIM = 100
			# tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
			# tokenizer.fit_on_texts(df['Consumer complaint narrative'].values)
			# word_index = tokenizer.word_index
			# print('Found %s unique tokens.' % len(word_index))


			def cnn():
				# Build CNN Model
				print('- Building CNN Model -')

				# Reshape dimension of data to fit CNN model
				X_train_res = np.expand_dims(X_train, axis=2)
				X_test_res = np.expand_dims(X_test, axis=2)
				print('Reshaped Data for Neural Networks: ', X_train_res.shape)

				# encode train class values as integers
				encoder = LabelEncoder()
				encoder.fit(y_train)
				encoded_y = encoder.transform(y_train)
				# convert integers to dummy variables (i.e. one hot encoded)
				y_train = np_utils.to_categorical(encoded_y)

				# encode test class values as integers
				encoder = LabelEncoder()
				encoder.fit(y_test)
				encoded_y = encoder.transform(y_test)
				# convert integers to dummy variables (i.e. one hot encoded)
				y_test = np_utils.to_categorical(encoded_y)


				model = Sequential()
				# input_shape = X_train_res[0].shape
				# print('Input shape: ', input_shape)
				# Input Layer
				model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(26,1)))
				# Dropout to prevent overfitting
				model.add(Dropout(0.25))
				# Hidden Layer 1
				model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
				model.add(Dropout(0.25))
				# Hidden Layer 2
				model.add(Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
				model.add(Dropout(0.25))
				# Flatten dimensions of data
				model.add(Flatten())
				# Output Layer
				model.add(Dense(num_classes, activation='softmax'))

				model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
				print(model.summary())

				# Fit Model
				model_train = model.fit(X_train_res, y_train, epochs=200, batch_size=batch_size, validation_data=(X_test_res, y_test))

				# model.save("'./Generated_Files/' + str(score_target) + '/Neural_Networks/CNN/model_2HL_100E_50BS.h5py")

				# Evaluate Model on Test Set
				print('Model Evaluation on Test Set')
				test_eval = model.evaluate(X_test_res, y_test, verbose=1)
				print('Test loss:', test_eval[0])
				print('Test accuracy:', test_eval[1])
				predicted_classes = model.predict(X_test_res)
				predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
				print(predicted_classes.shape, y_test.shape)
				correct = np.where(predicted_classes==y_test)[0]
				print("Found %d correct labels" % len(correct))

				for i, correct in enumerate(correct[:9]):
					print(predicted_classes[correct], test_Y[correct])
					# plt.subplot(3,3,i+1)
					# plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
					# plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
					# plt.tight_layout()

				display_plot = True
				if display_plot:
					accuracy = model_train.history['acc']
					val_accuracy = model_train.history['val_acc']
					loss = model_train.history['loss']
					val_loss = model_train.history['val_loss']
					epochs = range(len(accuracy))
					plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
					plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
					plt.title('Training and validation accuracy')
					plt.xlabel('Epochs')
					plt.ylabel('Accuracy')
					plt.legend()
					plt.savefig('./Generated_Files/' + str(score_target) + '/Multi_Class/CNN_Train_Val_Acc_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight')
					plt.show(block=False)
					plt.pause(0.5)
					plt.clf()
					plt.close()

					plt.plot(epochs, loss, 'bo', label='Training loss')
					plt.plot(epochs, val_loss, 'b', label='Validation loss')
					plt.title('Training and validation loss')
					plt.xlabel('Epochs')
					plt.ylabel('Loss')
					plt.legend()
					plt.savefig('./Generated_Files/' + str(score_target) + '/Multi_Class/CNN_Train_Val_Loss_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight')
					plt.show(block=False)
					plt.pause(0.5)
					plt.clf()
					plt.close()


			def lstm():
				print('Building LSTM model...')
				# from keras.utils import to_categorical
				# y_binary = to_categorical(y_int)

				# print(X_train)
				# print('Y TRAIN \n', y_train) # why are target values represented as 0 or 1?
				# print(y_train.reshape(y_train.shape[0], y_train.shape[1], 1))

				epoch, dropout = 20, 0.2
				print('EPOCH = ', epoch)
				print('DROPOUT = ', dropout)

				model = Sequential()
				model.add(Embedding(input_dim=1000, output_dim=100, input_length=num_features))
				model.add(SpatialDropout1D(0.2))
				model.add(LSTM(100))
				# model.add(RepeatVector(num_features))
				# model.add(LSTM(100, return_sequences=True))
				model.add(Dropout(dropout))
				model.add(Dense(num_classes+1, activation='softmax')) #had to add +1 as range was [0,5] so error when class no. 5 is found
				model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
				# model.compile(optimizer=RMSprop(lr=0.01), loss='categorical_crossentropy',metrics=['acc'])
				model.summary()

				# Train model
				history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1, validation_split=0.2)
				# history = model.fit(X_train, y_train.values.reshape(y_train.shape[0], y_train.shape[1], 1), epochs=epoch, batch_size=batch_size, verbose=1, validation_split=0.2)
				# Evaluate the model
				loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
				# loss, accuracy = model.evaluate(X_test_res, y_test.reshape(y_test.shape[0], y_test.shape[1], 1), verbose=1)
				print('Accuracy: %f' % (accuracy * 100))

				def display():
					plt.plot(history.history['acc'])
					plt.plot(history.history['val_acc'])

					plt.title('model accuracy')
					plt.ylabel('accuracy')
					plt.xlabel('epoch')
					plt.legend(['train','test'], loc = 'upper left')
					plt.show()

					plt.plot(history.history['loss'])
					plt.plot(history.history['val_loss'])

					plt.title('model loss')
					plt.ylabel('loss')
					plt.xlabel('epoch')
					plt.legend(['train','test'], loc = 'upper left')
					plt.show()
				# display()

			cnn()
			lstm()
		# neural_networks()
'''
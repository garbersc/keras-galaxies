from theano import config
import keras.callbacks
import keras.backend as T
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, MaxPooling1D, Dropout, Input, Convolution1D
from keras.layers.core import Lambda, Reshape
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

print config.optimizer

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        print 'batch ended'
	self.losses.append(logs.get('loss'))

class ValidLossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        print 'epoch ended'
	self.losses.append(model_noNorm.evaluate(X_test, Y_test, batch_size=32*5)[0])

n_data=int(1e5)

Y_train = np.random.binomial(n=1, p=0.3, size=(n_data,3))
Y_train = np.asarray(Y_train, dtype='float32')
X_train = np.random.randn(n_data,20)
X_train=X_train**2

Y_test = np.random.binomial(n=1, p=0.3, size=(32*5,3))
X_test = np.random.randn(32*5,20)
X_test=X_test**2

#model0 = Sequential()

#main_input = Input(shape=(None,10),batch_input_shape=(None,10), dtype='float32', name='main_input')
main_input = Input(batch_shape=(None,20) , dtype='float32', name='main_input')


#x=MaxPooling1D(2,input_shape=(20,2))(main_input)

#print x.shape


x=Dense(output_dim=40,  activation='relu',input_shape=(20,))(main_input)
x=Dropout(0.5)(x)

#x=Dense(output_dim=40, input_dim=10 activation='relu')#(main_input)
#model0.add(Activation("relu"))

x=Dense(output_dim=1024,  activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(output_dim=1024,  activation='relu')(x)
x=Dropout(0.5)(x)
'''
#model.add(MaxPooling1D())
model.add(Dense(output_dim=4000))
model.add(Activation("relu"))
model.add(Dropout(0.5))
#model.add(MaxPooling1D())
model.add(Dense(output_dim=4000))
model.add(Activation("relu"))
model.add(Dropout(0.5))
#model.add(MaxPooling1D())
model.add(Dense(output_dim=4000))
model.add(Activation("relu"))
model.add(Dropout(0.5))
#model.add(MaxPooling1D())
'''
x=Dense(output_dim=3,name='model0_out')(x)#,input_shape=(20,))(main_input)

#model0.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

'''
def lambdaFunction(x,normalise):
	if normalise: 
		print 'norm'
		x_ret=T.clip(x,0.0,0.001)
		#x_ret=T.ones_like(x)
	else:
		print 'no_norm'
		x_ret=x
	return T.reshape(x_ret,(x_ret.shape[0],3))

def output_shape(input_shape):
	return (input_shape[0],3)
'''

#l_noNorm=Lambda(lambdaFunction,output_shape,arguments={'normalise': False})(x)

#l_norm=Lambda(lambdaFunction,output_shape,arguments={'normalise': True})(x)

#model=Model(input=main_input,output=[l_noNorm,l_norm])
#model=Model(input=main_input,output=l_noNorm)

#model_norm=Model(input=main_input,output=l_norm)

model_noNorm=Model(input=main_input,output=x)

#print model_norm.input
#print model_norm.input_shape
#print model_norm.output_shape

#model_norm=Model(input=model0.get_layer('model0_out').output,output=l_norm)
	
#NORMALISE=T.variable(False)

#model_norm.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model_noNorm.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model_norm.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

#history = LossHistory()
#vHistory = ValidLossHistory()

model_noNorm.fit(X_train, Y_train, nb_epoch=5, batch_size=n_data)#, callbacks=[history,vHistory])

#history.on_train_begin()
#vHistory.on_train_begin()


#model=model_noNorm
#for i in xrange(0,3):
#	print "epoch %i/5" % (i+1)
#	for j in xrange(0,(X_train.shape[0])//10):
#		if j>0 or i>0:
#			model=model_norm
#			#NORMALISE.set_value(True)
#		#print NORMALISE
#		#print NORMALISE.get_value()
#		#print NORMALISE.eval()
#		print "%s/%s" %(j+1,(X_train.shape[0])//10)
#		#print T.shape( X_train[(j*(X_train.shape[0])//10) : ((j+1)*(X_train.shape[0])//10) ] )
#		#print T.shape( Y_train[ (j*(X_train.shape[0])//10) : ((j+1)*(X_train.shape[0])//10) ]  )
#		print model.train_on_batch( x=X_train[(j*(X_train.shape[0])//10) : ((j+1)*(X_train.shape[0])//10) ], y=Y_train[ (j*(X_train.shape[0])//10) : ((j+1)*(X_train.shape[0])//10) ] )
#		print model.predict_on_batch(x=X_train[(j*(X_train.shape[0])//10) : ((j+1)*(X_train.shape[0])//10) ])

		#history.on_batch_end(j,)
	#vHistory.on_epoch_end()

#print model_norm.get_weights()
#print model_noNorm.get_weights()
#print model_norm.get_weights()[0]==model_noNorm.get_weights()[0]
#loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32*5)



#print "\n"

#print loss_and_metrics

#lossplt = plt.plot(xrange(0,len(history.losses)),history.losses,'ro')
#lossplt = plt.plot(xrange(0,len(vHistory.losses)),vHistory.losses,'go')
#plt.show()

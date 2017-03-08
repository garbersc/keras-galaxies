import numpy as np
import keras.backend as T

def flat_galax_imgs(x):
	return x.reshape( ( x.shape[0] , x.shape[1]*x.shape[2]*x.shape[3] ) )

def rotate(x, rot_m, reduction = None ):
	if type(rot_m) == str:
		rot_m = np.load(rot_m)
	if not reduction: reduction = rot_m.shape[1]
	return T.dot( x , rot_m[:,:reduction]) 

def middle(x, mid_m):
	if type(mid_m) == str:
		mid_m = np.load(mid_m)
	return x - mid_m


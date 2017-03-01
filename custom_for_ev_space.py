import numpy as np

def flat_galax_imgs(x):
	if len(x.shape) >= 4:
		y = x.reshape(x.shape[0],np.prod(x.shape[1:-1]))
	else:	
		y = x.flatten()
	return y 

def rotate(x, rot_m, reduction = None ):
	if type(rot_m) = str:
		rot_m = np.load(rot_m)
	if not reduction: reduction = rot_m.shape[1]
	return: np.array([np.dot( p , rot_m[:,:reduction]) for p in x ]) 

def middle(x, mid_m):
	if type(mid_m) = str:
		mid_m = np.load(mid_m)
	return: x - mid_m


#main code from http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

import numpy as np
from numpy.linalg import eig, inv


def get_ellipse_kaggle_par(input_):

	x,y = points_from_input(input_,threshhold=1.)

	a = fitEllipse(x,y)
	ax_len = ellipse_axis_length(a)

	return {'axis_fraction' : (ax_len[0] / ax_len[1]) , 'ellipse_fit_goodness' : 0 }



def points_from_input(input_,threshhold,pointskip=1):
	x = []
	y = []
	for i in xrange(0,len(input_)):
		for j in xrange(0,len(input_[i])):
			if input_[i][j]>=threshhold:
				x.append(i)
				y.append(j)
	if pointskip>1:
		xn=[]
		yn=[]
		for i in xrange(0,len(x)):
			if not i%pointskip:
				xn.append(x[i])
				yn.append(y[i])
		x=xn
		y=yn

	return x,y
	

def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    #print a
    return a




def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return -0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    #if up< 0. or down1<=0. or down2<=0.: print up,down1,down2
    #epsilon=1e-10
    #if np.abs(up)<epsilon: up=0.
    #if np.abs(down1)<epsilon: down1=epsilon
    #if np.abs(down2)<epsilon: down2=epsilon
    res1=np.sqrt(np.abs(up/down1))
    res2=np.sqrt(np.abs(up/down2))
    return np.array([res1, res2])


def get_ellipse_par(input_,pointskip=1):

	x,y = points_from_input(input_,threshhold=1.,pointskip=pointskip)
	#print x
	#print y
	#print len(x)
	#print len(y)

	a = fitEllipse(np.asarray(x),np.asarray(y))
	#ax_len = ellipse_axis_length(a)
	
	#print a
	#print ellipse_axis_length(a)

	#print ellipse_center(a), ellipse_axis_length(a), ellipse_angle_of_rotation
	return_val = np.asarray([((ellipse_center(a)[0])),((ellipse_center(a)[1])),((ellipse_axis_length(a)[0])),((ellipse_axis_length(a)[1])),ellipse_angle_of_rotation(a)])
	return  return_val






#from theano import function, config, shared, tensor, sandbox
import keras.backend as T
import numpy
import time

print("---gpustyle:")

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = int(1e5)

rng = numpy.random.RandomState(22)
x = T.variable(numpy.asarray(rng.rand(vlen), dtype='float32'))
f = T.function([], T.exp(x))
#print(f.maker.fgraph.toposort())
print f
t0 = time.time()
for i in range(iters):
    r = f([])
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')




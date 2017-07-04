from theano import function, config, shared, tensor, sandbox
import theano.sandbox.cuda.basic_ops as sbcuda
import numpy
import time

print("Free GPU Mem %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))

vlen = 10 * 30 * 768   # 10 x #cores x # threads per core
iters = 1000

print('x')

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
print("Free GPU Mem %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))
for i in range(iters):
    r = f()
t1 = time.time()
print("Free GPU Mem %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))

print('y')

x = shared(numpy.asarray(rng.rand(vlen*1000), config.floatX))
t0 = time.time()
print("Free GPU Mem %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))
for i in range(iters):
    r = f()
t1 = time.time()
print("Free GPU Mem %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))
print("Looping %d times took %f seconds" % (iters, t1 - t0))

print("transfer y to cpu")
cy=x.transfer('cpu')
print("Free GPU Mem %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))
#print("transfer r to cpu")
#cr=r.transfer('cpu')
#print("Free GPU Mem %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))

print('r(y)')

x = r

t0 = time.time()
print("Free GPU Mem %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))
for i in range(iters):
    r = f()
t1 = time.time()
print("Free GPU Mem %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))
print("Looping %d times took %f seconds" % (iters, t1 - t0))


print('x2')

x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
t0 = time.time()
print("Free GPU Mem %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))
for i in range(iters):
    r = f()
t1 = time.time()
print("Free GPU Mem %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))

print("transfer y to gpu")
x=cy.transfer('gpu')
print("Free GPU Mem %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))

#r=cr.transfer('gpu')
#print("Free GPU Mem %s MiB " % (sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024.))

if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')


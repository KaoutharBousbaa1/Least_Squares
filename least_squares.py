## GENERATE AND PLOT ALL THE DATA

import numpy as np
import matplotlib.pyplot as plt

rng(0)
N = 15
x = np.transpose(np.linspace(0,5,N))
y = (x - 2) ** 2 + np.random.randn(N,1)
# polynomial of order k
k = 5
# compute Least Squares fit
A = []
for i in np.arange(k,0+- 1,- 1).reshape(-1):
    A = np.array([A,x ** i])

what = np.linalg.solve(A,y)
# evaluate the polynomial on evenly spaced points.
# NOTE: you can also use the command yy = polyval(what,xx);
xx = np.transpose(np.linspace(0,5,200))
AA = []
for i in np.arange(k,0+- 1,- 1).reshape(-1):
    AA = np.array([AA,xx ** i])

yy = AA * what
# plot the data points and the best fit
plt.figure(1)
plt.plot(x,y,'b.',xx,yy,'k-','MarkerSize',20)
err = norm(y - A * what)
plt.title(np.array(['best fit using a polynomial of degree ',num2str(k)]))
## NO CROSS-VALIDATAION

kmax = 5

# generate one large A matrix. We can get the matrices for particular k
# values by just picking the correct columns from this matrix.
A = []
for i in np.arange(kmax,0+- 1,- 1).reshape(-1):
    A = np.array([A,x ** i])

err = np.zeros((kmax,1))
for k in np.arange(1,kmax+1).reshape(-1):
    kvals = np.arange(kmax - k + 1,kmax + 1+1)
    Amat = A[:,kvals]
    what = np.linalg.solve(Amat,y)
    # compute the error and divide by the number of points
    err[k] = norm(y - Amat * what) / N

## USIGN CROSS-VALIDATION

rng(0)
T = 12

trials = 1000

# generate one large A matrix. We can get the matrices for particular k
# values by just picking the correct columns from this matrix.
A = []
for i in np.arange(kmax,0+- 1,- 1).reshape(-1):
    A = np.array([A,x ** i])

errcv = np.zeros((kmax,trials))
for t in np.arange(1,trials+1).reshape(-1):
    r = randperm(N)
    train = r(np.arange(1,T+1))
    test = r(np.arange(T + 1,end()+1))
    for k in np.arange(1,kmax+1).reshape(-1):
        kvals = np.arange(kmax - k + 1,kmax + 1+1)
        Atrain = A(train,kvals)
        Atest = A(test,kvals)
        ytrain = y(train)
        ytest = y(test)
        what = np.linalg.solve(Atrain,ytrain)
        # compute error and divide by the number of test points
        errcv[k,t] = norm(ytest - Atest * what) / (N - T)

avg_err_cv = mean(errcv,2)
# compare the performance of the least-squares on training data to the
# performance when using cross-validation
plt.figure(2)
clf
bar(np.array([err,avg_err_cv]))
plt.xlabel('polynomial degree')
plt.ylabel('average test error')
plt.legend('performance on training data','using cross-validation','Location','north')
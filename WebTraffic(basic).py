import scipy as sp
import matplotlib.pyplot as plt

data = sp.genfromtxt("web_traffic.tsv.txt", delimiter="\t")#input data
print(data[:10])#Data till 10nth row
print(data.shape)#(743, 2)=(no. of samples,No. of features of samples)

x = data[:,0]#Choosing column 1
y = data[:,1]#Choosing column 2

sp.sum(sp.isnan(y))#Total Nan values

#Columns without Nan values
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

def error(f, x, y):
 return sp.sum((f(x)-y)**2)

plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],
 ['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()

fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
#fp1,residuals,rank,sv,rcond
#(array([2.59619213, 989.02487106]),array([3.17389767e+08]),2,array([1.36699071,0.36240365]), 1.6320278461989801e-13)

print("Model parameters: %s" % fp1)
print(residuals)#Approximate error

f1 = sp.poly1d(fp1)# 2.596 x + 989
print(error(f1, x, y))

fx = sp.linspace(0,x[-1],1000) # generate X-values for plotting. 1000 is for having 1000 values from 0-743
plt.plot(fx, f1(fx), linewidth=4)
plt.legend(["d=%i" % f1.order], loc="upper left")

f2p = sp.polyfit(x, y, 2)
print(f2p)
f2 = sp.poly1d(f2p)
print(error(f2, x, y))

fx = sp.linspace(0,x[-1],1000) # generate X-values for plotting. 1000 is for having 1000 values from 0-743
plt.plot(fx, f2(fx), linewidth=4)
plt.legend(["d=%i" % f2.order], loc="upper left")
plt.show()

import csv

import  numpy as np
import matplotlib.pyplot as plt
m1_mean = [1, 1]
m1_cov = [[2, 0], [0, 2]]
m1 = np.random.multivariate_normal(m1_mean, m1_cov, 333)

#print(m1)
print(np.mean(m1,axis=0))
print(np.cov(m1.T))



m2_mean = [4, 4]
m2_cov = [[2, 0], [0, 2]]
m2 = np.random.multivariate_normal(m2_mean, m2_cov, 333)
#print(m2)
print(np.mean(m2,axis=0))
print(np.cov(m2.T))

m3_mean = [8, 1]
m3_cov = [[2, 0], [0, 2]]
m3 = np.random.multivariate_normal(m3_mean, m3_cov, 334)
#print(m3)
print(np.mean(m3,axis=0))
print(np.cov(m3.T))

f=open('dataset1.csv','w',newline='')
writer=csv.writer(f)
a = np.ones(333)
b = np.ones(333)
for i in range(0,333):
    b[i]=2
c = np.ones(334)
for i in range(0,334):
    c[i]=3

for i in np.c_[m1,a]:
    writer.writerow(i)
for i in np.c_[m2,b]:
    writer.writerow(i)
for i in np.c_[m3,c]:
    writer.writerow(i)

fig = plt.figure(0)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset X1')
plt.scatter(m1[:,0], m1[:,1], c='red', alpha=1, marker='+', label='m1') # c='red'定义为红色，alpha是透明度，marker是画的样式
plt.scatter(m2[:,0], m2[:,1], c='green', alpha=1, marker='+', label='m2')
plt.scatter(m3[:,0], m3[:,1], c='blue', alpha=1, marker='+', label='m3')
plt.grid(True)
plt.legend(loc='best')
plt.show()




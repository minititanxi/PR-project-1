import csv
import math

import  numpy as np


def likelihood(x,u1,u2,u3,pr_w1,pr_w2,pr_w3):#似然率测试规则
    co_var = np.mat([[2, 0], [0, 2]])
    num = np.array([0,0,0])
    t1 = np.mat(x - u1)
    t2 = np.mat(x - u2)
    t3 = np.mat(x - u3)
    #print(t1)
    #print(co_var.I)
    #print(t1*co_var.I)
    #print(t1*co_var.I*t1.T)
    #print(np.linalg.det(t1*co_var.I*t1.T))
    if (-0.5)*np.linalg.det(t1*co_var.I*t1.T)>math.log(pr_w2/pr_w1)+(-0.5)*np.linalg.det(t2*co_var.I*t2.T):
        num[0] = num[0] + 1
    else:
        num[1] = num[1] + 1
    if (-0.5)*np.linalg.det(t1*co_var.I*t1.T)>math.log(pr_w3/pr_w1)+(-0.5)*np.linalg.det(t3*co_var.I*t3.T):
        num[0] = num[0] + 1
    else:
        num[2] = num[2] + 1
    if (-0.5)*np.linalg.det(t2*co_var.I*t2.T)>math.log(pr_w3/pr_w2)+(-0.5)*np.linalg.det(t3*co_var.I*t3.T):
        num[1] = num[1] + 1
    else:
        num[2] = num[2] + 1
    #print(num)
    #print(np.where(num == np.max(num))[0])
    return np.where(num == np.max(num))[0]

def Bayes(x,u1,u2,u3,pr_w1,pr_w2,pr_w3,C):#贝叶斯风险规则
    co_var = np.mat([[2, 0], [0, 2]])
    num = np.array([0, 0, 0])
    t1 = np.mat(x - u1)
    t2 = np.mat(x - u2)
    t3 = np.mat(x - u3)
    #print(t1)
    #print(co_var.I)
    #print(t1 * co_var.I)
    #print(t1 * co_var.I * t1.T)
    #print(np.linalg.det(t1 * co_var.I * t1.T))
    if (-0.5) * np.linalg.det(t1 * co_var.I * t1.T) > math.log((pr_w2 / pr_w1)*((C[0][1]-C[1][1])/(C[1][0]-C[0][0]))) + (-0.5) * np.linalg.det(
            t2 * co_var.I * t2.T):
        num[0] = num[0] + 1
    else:
        num[1] = num[1] + 1
    if (-0.5) * np.linalg.det(t1 * co_var.I * t1.T) > math.log((pr_w3 / pr_w1)*((C[0][2]-C[2][2])/(C[2][0]-C[0][0]))) + (-0.5) * np.linalg.det(
            t3 * co_var.I * t3.T):
        num[0] = num[0] + 1
    else:
        num[2] = num[2] + 1
    if (-0.5) * np.linalg.det(t2 * co_var.I * t2.T) > math.log((pr_w3 / pr_w2)*((C[1][2]-C[2][2])/(C[2][1]-C[1][1])))  + (-0.5) * np.linalg.det(
            t3 * co_var.I * t3.T):
        num[1] = num[1] + 1
    else:
        num[2] = num[2] + 1
    #print(num)
    #print(np.where(num == np.max(num))[0])
    return np.where(num == np.max(num))[0]

def min_distance(x,u1,u2,u3):#最短欧式距离规则
    num = np.array([0.0, 0.0, 0.0])
    t1 = x - u1
    t2 = x - u2
    t3 = x - u3
    num[0] = t1[0]*t1[0]+t1[1]*t1[1]
    num[1] = t2[0]*t2[0]+t2[1]*t2[1]
    num[2] = t3[0]*t3[0]+t3[1]*t3[1]
    return np.where(num == np.min(num))[0]




csv_reader = csv.reader(open('dataset2.csv', encoding='utf-8'))
u1 = np.array([1,1])
u2 = np.array([4,4])
u3 = np.array([8,1])
sum=0
for row in csv_reader:#似然率测试规则
    x = np.array([float(row[0]),float(row[1])])
    #print(x)
    pr_w1 = 0.6
    pr_w2 = 0.3
    pr_w3 = 0.1
    predict=likelihood(x, u1, u2, u3, pr_w1, pr_w2, pr_w3)
    #print(predict[0])
    if float(row[2])==predict[0]+1:
        sum=sum+1
print(sum/1000)


csv_reader = csv.reader(open('dataset2.csv', encoding='utf-8'))
#贝叶斯风险规则
Cij=np.array([[0,2,3],[1,0,2.5],[1,1,0]])
sum=0
for row in csv_reader:
    x = np.array([float(row[0]),float(row[1])])
    #print(x)
    pr_w1 = 0.6
    pr_w2 = 0.3
    pr_w3 = 0.1
    predict=Bayes(x, u1, u2, u3, pr_w1, pr_w2, pr_w3,Cij)
    #print(predict[0])
    if float(row[2]) == predict[0] + 1:
        sum = sum + 1
print(sum / 1000)

#最大后验概率规则
csv_reader = csv.reader(open('dataset2.csv', encoding='utf-8'))
Cij=np.array([[0,1,1],[1,0,1],[1,1,0]])
sum=0
for row in csv_reader:
    x = np.array([float(row[0]),float(row[1])])
    #print(x)
    pr_w1 = 0.6
    pr_w2 = 0.3
    pr_w3 = 0.1
    predict=Bayes(x, u1, u2, u3, pr_w1, pr_w2, pr_w3,Cij)
    #print(predict[0])
    if float(row[2]) == predict[0] + 1:
        sum = sum + 1
print(sum / 1000)

#最短欧式距离准则
csv_reader = csv.reader(open('dataset2.csv', encoding='utf-8'))
sum=0
for row in csv_reader:
    x = np.array([float(row[0]),float(row[1])])
    #print(x)
    pr_w1 = 0.6
    pr_w2 = 0.3
    pr_w3 = 0.1
    predict=min_distance(x, u1, u2, u3)
    #print(predict[0])
    if float(row[2]) == predict[0] + 1:
        sum = sum + 1
print(sum / 1000)









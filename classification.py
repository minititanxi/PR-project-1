import csv
import math

import  numpy as np


def likelihood(x,u1,u2,u3,pr_w1,pr_w2,pr_w3):
    co_var= np.array([[2, 0], [0, 2]])
    num=np.array([0,0,0])
    if (-0.5)*np.linalg.det((x-u1)*co_var*(x-u1).T)>math.log(pr_w2/pr_w1)+(-0.5)*np.linalg.det((x-u2)*co_var*(x-u2).T):
        num[0] = num[0] + 1
    else:
        num[1] = num[1] + 1
    if (-0.5)*np.linalg.det((x-u1)*co_var*(x-u1).T)>math.log(pr_w3/pr_w1)+(-0.5)*np.linalg.det((x-u3)*co_var*(x-u3).T):
        num[0] = num[0] + 1
    else:
        num[2] = num[2] + 1
    if (-0.5)*np.linalg.det((x-u2)*co_var*(x-u2).T)>math.log(pr_w3/pr_w2)+(-0.5)*np.linalg.det((x-u3)*co_var*(x-u3).T):
        num[1] = num[1] + 1
    else:
        num[2] = num[2] + 1
    print(np.where(num == np.max(num))[0])
    return




csv_reader = csv.reader(open('dataset1.csv', encoding='utf-8'))
u1 = np.array([1,1])
u2 = np.array([4,4])
u3 = np.array([8,1])
for row in csv_reader:
    x = np.array([float(row[0]),float(row[1])])
    #print(x)
    pr_w1 = (1 / 3)
    pr_w2 = (1 / 3)
    pr_w3 = (1 / 3)
    likelihood(x, u1, u2, u3, pr_w1, pr_w2, pr_w3)







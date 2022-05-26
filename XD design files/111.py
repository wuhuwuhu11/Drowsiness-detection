 # coding=gbk
#python edition: Python3.4.1,2014,10,17
import numpy as np
from numpy import linalg as la

def read_points():
    dataset=[]
    with open('1.txt','r') as file:
        for line in file:
            if line =='\n':
                continue
            dataset.append(list(map(float,line.split(' '))))
        file.close()
        return  dataset

def generate_traineddata():
    arr=[[] for i in range(3)]
    with open('2.txt','r') as file:
        index=0
        for line in file:
            if line=='\n' :
                continue
            elif line[0]=='C':
                index=int(line[-2])-1
                continue
            arr[index].append(int(line))
        file.close()
    train=[[] for i in range(3)]
    test=[[] for i in range(3)]
    for i in range(len(arr)):
        tr=int(0.67* len(arr[i]))
        train[i]=arr[i][:tr]
        test[i]=arr[i][tr:]
    f1=open('trained.txt','w')
    f2=open('tested.txt','w')
    print(train,end='\n')
    print(test,end='\n')
    for i in range(3):
        for j in train[i]:
            f1.write("%d\n"%j)
        f1.write('\n')
        for k in test[i]:
            f2.write("%d\n"%k)
        f2.write('\n')
    f1.close()
    f2.close()
    return train,test

def createMatrix(train,test,dataset):
    trainmat=[[] for i in range(3)]
    testmat=[[] for i in range(3)]
    for i in range(3):
        for j in train[i]:
            trainmat[i].append(dataset[j])
        for k in test[i]:
            testmat[i].append(dataset[k])
    return   trainmat,testmat

def classify(trainmat,testmat,test):
    #求三类训练集的均值向量
    tr1,tr2,tr3=np.mat(trainmat[0]),np.mat(trainmat[1]),np.mat(trainmat[2])
    te=[[] for i in range(3)]
    te[0],te[1],te[2]=np.mat(testmat[0]),np.mat(testmat[1]),np.mat(testmat[2])
    u01=np.mean(tr1,axis=0)
    u02=np.mean(tr2,axis=0)
    u03=np.mean(tr3,axis=0)
    #获得矩阵长度
    l1,l2,l3=len(trainmat[0]),len(trainmat[1]),len(trainmat[2])
    #求三类训练集的类内离散度矩阵
    s1,s2,s3=0,0,0
    for i in range(l1):
        s1=s1+(tr1[i]-u01).T*(tr1[i]-u01)
    for i in range(l2):
        s2=s2+ (tr2[i]-u02).T*(tr2[i]-u02)
    for i in range(l3):
        s3=s3+ (tr3[i]-u03).T*(tr3[i]-u03)
    #总类内离散度矩阵
    sw12,sw13,sw23=s1+s2,s1+s3, s2+s3
    #求向量W*与边界
    W12=la.inv(sw12)*((u01-u02).T)
    W012=(l1*(W12.T)*(u01.T)+l2*(W12.T)*(u02.T))/(l1+l2)
    W13=la.inv(sw13)*((u01-u03).T)
    W013=(l1*(W13.T)*(u01.T)+l3*(W13.T)*(u03.T))/(l1+l3)
    W23=la.inv(sw23)*(u02-u03).T
    W023=(l2*(W23.T)*(u02.T)+l3*(W23.T)*(u03.T))/(l3+l2)
    result=[[] for i in range(4)]
    for i in range(3):
        testset=te[i]
        count=0
        for X in testset:
            if ((W12.T)*(X.T)-W012>0) and   ((W13.T)*(X.T)-W013>0):
                result[0].append(test[i][count])
            elif  ((W12.T)*(X.T)-W012<0) and   ((W23.T)*(X.T)-W023>0):
                result[1].append(test[i][count])
            elif    ((W13.T)*(X.T)-W013<0) and   ((W23.T)*(X.T)-W023<0):
                result[2].append(test[i][count])
            else:
                result[3].append(test[i][count])
            count=count+1
    str1='类内离散度矩阵：'
    str2='投影方向：'
    str3='边界点：'
    str4='Fisher得出的分类结果：'
    fisher=open('fisher.txt','w')
    print(str1,'s1:',end='\n',file=fisher)
    print(s1,end='\n',file=fisher)
    print(str1,'s2:',end='\n',file=fisher)
    print(s2,end='\n',file=fisher)
    print(str1,'s3:',end='\n',file=fisher)
    print(s3,end='\n',file=fisher)
    print(str2,'\n','W12:\n',W12,end='\n',file=fisher)
    print('W13:\n',W13,end='\n',file=fisher)
    print('W23:\n',W23,end='\n',file=fisher)
    print(str3,'\n','W012:',W012,end='\n' ,file=fisher)
    print('W013:',W013,end='\n' ,file=fisher)
    print('W023:',W023,end='\n' ,file=fisher)
    print(str4,end='\n' ,file=fisher)
    for i in range(4):
        print('第%d类'%(i+1),result[i],end='\n' ,file=fisher)
    fisher.close()
    return    result


def main():
    dataset=read_points()
    train,test= generate_traineddata()
    trainmat,testmat=createMatrix(train,test,dataset)
    result= classify(trainmat,testmat,test)
    print(result)

if __name__=='__main__':
    main()
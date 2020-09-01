# mengimport pandas, numpy, seaborn, pyplot, random, math dari library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as sklearnm
import random as rnd
import math
import time


#memulai menghitung waktu yang dibutuhkan untuk eksekusi
startt = time.clock()

# membaca data pada text file
data = pd.read_csv("level3_asg2_konv.csv", sep=",", header=None)

# memberikan nama kolom
namafitur = ["lama", "langkah", "set", "remove", "C1", "C2", "C3", "C4", "C5", "C6", "unik", "err"]
data.columns = namafitur
datautuh = data

# menormalisasi dataset
normalizeData = (data - data.min()) / (data.max() - data.min())
normalizeData = normalizeData.fillna(0)

# inisialisasi data kedalam bentuk matiks
data = (normalizeData.values.tolist())
# print(*data, sep='\n')

# inisialisasi pso
jml_cluster = 3 #k
jml_partikel = 20
intbobot = [0.4, 0.9]  # w
c1=0.5
c2=0.5
intkecepatan = [-0.5, 0.5]  # v
jml_iterasi = 150
r1 = rnd.random()
r2 = rnd.random()
fitur = len(data[0])

#inisialisasi partikel awal
partikel = [[] for j in range(jml_partikel)]
for i in range(jml_partikel):
    x = [k for k in range(len(data))]
    rnd.shuffle(x)
    for j in range(jml_cluster):
        y = x[j]
        partikel[i].append(data[y])

def hitungJarakPSO(partikel):
    jarak = [ [[] for i in range(jml_cluster)] for j in range(jml_partikel)]
    for p in range (jml_partikel):
        for x in range (jml_cluster) :
            for y in range (len(data)) :
                temp=0
                for z in range (len(data[1])) :
                    temp += (partikel[p][x][z]-data[y][z])**2
                temp = math.sqrt(temp)
                jarak[p][x].append(temp)
    return jarak

def minclusPSO(jarak):
    mymin=[[] for p in range(jml_partikel)]
    for i in range (jml_partikel):
        jarak[i] = [list(j) for j in zip(*jarak[i])]
        for x in jarak[i] :
            mymin[i].append(min(x))
    return mymin

def pengelompokanPSO(jarak,mymin): #menentukan cluster
    c=[[] for p in range(jml_partikel)]
    for i in range (jml_partikel):
        #jarak[i] = [list(j) for j in zip(*jarak[i])]
        for x in (range(len(jarak[i]))):
            for y in range (jml_cluster):
                if (jarak[i][x][y]== mymin[i][x]) :
                    c[i].append(y+1)
    return c

def rataratacluster(cluster,jarak): #mencari nilai fitness
    fitness=[]
    ratarata=[[] for i in range(jml_partikel)]

    for x in range(jml_partikel):
        jarak[x] = [list(j) for j in zip(*jarak[x])] #percluster
        temp=0
        for y in range (jml_cluster):
            k=0
            tempp=0
            for z in range(len(data)):
                if(cluster[x][z]==(y+1)):
                    tempp += jarak[x][y][z]
                    k+=1
            if (k!=0):
                ratarata[x].append(tempp/k)
            else:
                ratarata[x].append(0)
            temp+=ratarata[x][y]
        fitness.append(temp/jml_cluster)
    return (fitness)

def partikelgbest(fitness,pbest):
    gbest=min(fitness)
    #print(gbest)
    for i in range(jml_partikel):
        if(gbest==fitness[i]):
            pgbest=pbest[i]
    return pgbest

### iterasi-0 ###
jarakPSO = hitungJarakPSO(partikel)
print(jarakPSO)
myminPSO = minclusPSO(jarakPSO)
print(myminPSO)
clusterbaruPSO = pengelompokanPSO(jarakPSO,myminPSO)
print(clusterbaruPSO)
fitness=rataratacluster(clusterbaruPSO,jarakPSO)

#inisialisasi kecepatan awal
kecepatan=[ [[] for i in range(jml_cluster)] for j in range(jml_partikel)]
for x in range(jml_partikel):
    for y in range(jml_cluster):
        for z in range(fitur):
            kecepatan[x][y].append(0)
pbest=partikel
gbest=partikelgbest(fitness,pbest)

def updatekecepatan(w,kecepatan,pbest,partikel,gbest):
    for x in range(jml_partikel):
        for y in range(jml_cluster):
            for z in range(fitur):
                temp=(w*kecepatan[x][y][z])+(c1*r1*(pbest[x][y][z]-partikel[x][y][z]))+(c2*r2*(gbest[y][z]-partikel[x][y][z]))
                if (temp<=intkecepatan[0]):
                    kecepatan[x][y][z]=intkecepatan[0]
                elif (temp>=intkecepatan[1]):
                    kecepatan[x][y][z]=intkecepatan[1]
                else:
                    kecepatan[x][y][z]=temp
    return(kecepatan)
def updateposisi(partikel,kecepatan):
    partikelnew=[ [[] for i in range(jml_cluster)] for j in range(jml_partikel)]
    for x in range(jml_partikel):
        for y in range(jml_cluster):
            for z in range(fitur):
                if((partikel[x][y][z]+kecepatan[x][y][z])<=0):
                    partikelnew[x][y].append(0)
                elif((partikel[x][y][z]+kecepatan[x][y][z])>=1):
                    partikelnew[x][y].append(1)
                else:
                    partikelnew[x][y].append(partikel[x][y][z]+kecepatan[x][y][z])
    return(partikelnew)

def cekfitness(fitness,fitnessnew):
    minfitness=[]
    for i in range(len(fitness)):
        minfitness.append(min(fitness[i],fitnessnew[i]))
    return (minfitness)
def updatepbest(fitnessnew,minfitness,partikel,pbest):
    for x in range(jml_partikel):
        for y in range(jml_cluster):
            for z in range(fitur):
                if(minfitness[x]==fitnessnew[x]):
                    pbest[x]=partikel[x]
    return (pbest)

### iterasi-n ###
for i in range(jml_iterasi):
    w=(intbobot[1]-((intbobot[1]-intbobot[0])/jml_iterasi)*(i+1))
    kecepatan = updatekecepatan(w,kecepatan,pbest,partikel,gbest)
    partikel = updateposisi(partikel,kecepatan)
    jarakPSO = hitungJarakPSO(partikel)
    myminPSO = minclusPSO(jarakPSO)
    clusterPSO = pengelompokanPSO(jarakPSO,myminPSO)
    fitnessnew=rataratacluster(clusterPSO,jarakPSO)
    minfitness=cekfitness(fitness,fitnessnew)
    pbest=updatepbest(fitnessnew,minfitness,partikel,pbest)
    gbest=partikelgbest(minfitness,pbest)
    fitness=minfitness

#inisialisasi kmeans
maxLoops = 100
centroid=gbest


# Perhitungan jarak pusat cluster dengan rumus euclidean distance
def minclus(jarak):
    mymin = []
    jarak = [list(i) for i in zip(*jarak)]
    for x in jarak:
        mymin.append(min(x))
    return mymin

def hitungJarak(centroid):
    c = [[] for i in range(jml_cluster)]
    for x in range(jml_cluster):
        for y in range(len(data)):
            temp = 0
            for z in range(len(data[1])):
                temp += (centroid[x][z] - data[y][z]) ** 2
            temp = math.sqrt(temp)
            c[x].append(temp)
    return c

def pengelompokan(jarak, mymin):  # menentukan cluster
    c = []
    jarak = [list(i) for i in zip(*jarak)]
    for x in (range(len(jarak))):
        for y in range(jml_cluster):
            if (jarak[x][y] == mymin[x]):
                c.append(y + 1)
                break;
    return c

def pusatCluster(cluster): #mengupdate pusat cluster
    centroid=[[] for i in range(jml_cluster)]
    for x in range(fitur):
        for y in range(jml_cluster):
            k=0
            temp=0
            for z in range(len(data)):
                if (cluster[z]==(y+1)):
                    temp+=data[z][x]
                    k+=1
            if (k!=0):
                centroid[y].append(temp/k)
            else:
                centroid[y].append(0)
    return centroid

def cekIterasi(clusterlama,clusterbaru,i):
    x=0
    for z in range(len(clusterbaru)):
        if(clusterlama[z] != clusterbaru[z]):
            x+=1
    if (x==0):
        i+=maxLoops
    return i

#melakukan iterasi
clusterlama=[]
for x in range(len(data)):
    clusterlama.append(0)
i=0
while(i<=maxLoops):
    i+=1
    print('iterasi ke -',(i))
    jarak = hitungJarak(centroid)
    mymin = minclus(jarak)
    clusterbaru = pengelompokan(jarak,mymin)
    i = cekIterasi(clusterlama,clusterbaru,i)
    clusterlama = clusterbaru
    centroid.clear()
    centroid = pusatCluster(clusterbaru)
    print(clusterbaru)
    #i+=1
    print()
datautuh['cluster']=clusterbaru
print(datautuh)


endd=time.clock()

#menghitung Silhouette Coefficient
#euclidean tiap data
euc = [ [] for i in range(len(data))]
for x in range (len(data)) : #20 loop
    for y in range (len(data)) : #20 loop
        temp=0
        for z in range (len(data[1])) : #12 loop
            temp += (data[x][z]-data[y][z])**2
        temp = math.sqrt(temp)
        euc[x].append(temp)

#perhitungan ai
print(clusterbaru)
ai=[]
for x in range (len(clusterbaru)) : # loop 20
    temp=0
    n=0
    for y in range(len(clusterbaru)) : #loop 20
        if (clusterbaru[x]==clusterbaru[y]):
            temp+=euc[x][y]
            n+=1
    ai.append(temp/n)
#print(ai)

#perhitungan bi
d=[ [] for i in range(len(data))]
bi=[]

for x in range (len(clusterbaru)) : # loop 20
    for y in range(jml_cluster) :
        temp=0
        n=0
        for z in range(len(clusterbaru)) : #loop 20
            if ((y+1)!=clusterbaru[x]):
                if(clusterbaru[z]==(y+1)):
                    #print('euc',euc[x][j])
                    temp+=euc[x][z]
                    n+=1
        if(temp!=0):
            d[x].append(temp/n)
    bi.append(min(d[x]))

#perhitungan si
si=[]
maxab=[]
sigsi=0
for x in range(len(ai)):
    maxab.append(max(ai[x],bi[x]))
    si.append((bi[x]-ai[x])/maxab[x])
    sigsi+=si[x]
#print(maxab)
#print(si)
sigsi=(sigsi/len(si))
print(sigsi)

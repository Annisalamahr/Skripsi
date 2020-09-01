# mengimport pandas, numpy, seaborn, pyplot, random, math dari library
import pandas as pd
import random as rnd
import math
import time

startt = time.clock()

# membaca data pada text file
data = pd.read_csv("level3_asg2_konv.csv", sep=",", header=None)

# memberikan nama kolom
namafitur =["lama", "langkah", "set", "remove", "C1", "C2", "C3", "C4", "C5", "C6", "unik", "err"]
data.columns = namafitur
datautuh = data
print(data)

# menormalisasi dataset
columnLength = len(data.columns)
normalizeData = (data - data.min()) / (data.max() - data.min())
datanorm=normalizeData
print(normalizeData)

# inisialisasi data kedalam bentuk matiks
data =(normalizeData.values.tolist())

# inisialisasi kmeans
jml_cluster=3
maxLoops = 100
fitur = len(data[0])

centroid = []
for j in range(jml_cluster):
    x = []
    for k in range(fitur):
        x.append(rnd.random())
    centroid.append(x)

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

def pusatCluster(cluster):  # mengupdate pusat cluster
    centroid = [[] for i in range(jml_cluster)]
    for x in range(fitur):
        for y in range(jml_cluster):
            k = 0
            temp = 0
            for z in range(len(data)):
                if (cluster[z] == (y + 1)):
                    temp += data[z][x]
                    k += 1
            if (k != 0):
                centroid[y].append(temp / k)
            else:
                centroid[y].append(0)
    return centroid

def cekIterasi(clusterlama, clusterbaru, i):
    x = 0
    for z in range(len(clusterbaru)):
        if (clusterlama[z] != clusterbaru[z]):
            x += 1
    if (x == 0):
        i += maxLoops
    return i

# melakukan iterasi
clusterlama = []
for x in range(len(data)):
    clusterlama.append(0)
i = 0
while (i <= maxLoops):
    i += 1
    jarak = hitungJarak(centroid)
    mymin = minclus(jarak)
    clusterbaru = pengelompokan(jarak, mymin)
    i = cekIterasi(clusterlama, clusterbaru, i)
    clusterlama = clusterbaru
    centroid.clear()
    centroid = pusatCluster(clusterbaru)
datautuh['cluster'] = clusterbaru

endd = time.clock()

# menghitung Silhouette Coefficient
# euclidean tiap data
euc = [[] for i in range(len(data))]
for x in range(len(data)):  # 20 loop
    for y in range(len(data)):  # 20 loop
        temp = 0
        for z in range(len(data[1])):  # 12 loop
            temp += (data[x][z] - data[y][z]) ** 2
        temp = math.sqrt(temp)
        euc[x].append(temp)

# perhitungan ai
ai = []
for x in range(len(clusterbaru)):  # loop 20
    temp = 0
    n = 0
    for y in range(len(clusterbaru)):  # loop 20
        if (clusterbaru[x] == clusterbaru[y]):
            temp += euc[x][y]
            n += 1
    ai.append(temp / n)
# print(ai)

# perhitungan bi
d = [[] for i in range(len(data))]
bi = []

for x in range(len(clusterbaru)):  # loop 20
    for y in range(jml_cluster):
        temp = 0
        n = 0
        for z in range(len(clusterbaru)):  # loop 20
            if ((y + 1) != clusterbaru[x]):
                if (clusterbaru[z] == (y + 1)):
                    # print('euc',euc[x][j])
                    temp += euc[x][z]
                    n += 1
        if (temp != 0):
            d[x].append(temp / n)
    bi.append(min(d[x]))

# perhitungan si
si = []
maxab = []
sigsi = 0
for x in range(len(ai)):
    maxab.append(max(ai[x], bi[x]))
    si.append((bi[x] - ai[x]) / maxab[x])
    sigsi += si[x]
sigsi = (sigsi / len(si))
print(sigsi)

waktu=(endd - startt)

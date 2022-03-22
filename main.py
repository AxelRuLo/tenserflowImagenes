import os
import re
import matplotlib.pyplot as plt
import mainDensa
import mainDensaDos
import mainConvulucional

dirname = os.path.join(os.getcwd(), "deportes")
imgpath = dirname + os.sep

images = []
directories = []
dircount = []
prevRoot = ""
cant = 0

print("leyendo imagenes de ", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant = cant + 1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            images.append(image)
            b = "Leyendo..." + str(cant)
            print(b, end="\r")
            if prevRoot != root:
                print(root, cant)
                prevRoot = root
                directories.append(root)
                dircount.append(cant)
                cant = 0
dircount.append(cant)
dircount = dircount[1:]
dircount[0] = dircount[0] + 1

densa1 = mainDensa.densa1(images,directories,dircount,prevRoot,cant)
densa2 = mainDensaDos.densa2(images,directories,dircount,prevRoot,cant)
convolucional = mainConvulucional.convolucionar(images,directories,dircount,prevRoot,cant)

arrayResultados = [densa1,densa2,convolucional]

plt.xlabel("epoch")
plt.ylabel("error")
plt.plot(arrayResultados[0],color="blue",)
plt.plot(arrayResultados[1],color="green",)
plt.plot(arrayResultados[2],color="red")
plt.legend(["modelado densa 4 capas","modelado densa 6 capas","modelado convolucional"]) 
plt.show()
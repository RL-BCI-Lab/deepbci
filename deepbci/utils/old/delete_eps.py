import os,sys
path = str(sys.argv[1]) # path to folder with .eps files make sure to add / at the end
c= 1

def sort_(alist):
    return sorted(alist, key=embedded_numbers)

for file in os.listdir(path):
    if file.endswith(".eps"):
         print(str(c), end="\r")
         os.remove(path+file)
         c += 1

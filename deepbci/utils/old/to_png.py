from wand.image import Image
import os,re,sys

re_digits = re.compile(r'(\d+)')
path = str(sys.argv[1]) # path to folder with .eps files make sure to add / at the end

def embedded_numbers(s):
    pieces = re_digits.split(s)             # split into digits/nondigits
    pieces[1::2] = map(int, pieces[1::2])   # turn digits into numbers
    return pieces

def sort_(alist):
    return sorted(alist, key=embedded_numbers)

eps_list = []
for file in os.listdir(path):
    if file.endswith(".eps"):
        eps_list.append(file)

eps_list = sort_(eps_list)
print(path+eps_list[1])

for i in range(len(eps_list)):
    print(str(i) + " " + str(len(eps_list)), end="\r")
    filename=path+eps_list[i]
    with Image(filename=filename, format='png') as img:
         img.save(filename=path+"state"+str(i+1)+".png")

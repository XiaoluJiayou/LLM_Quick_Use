import glob
import os

base_dir = r"C:\Users\unicom350\Desktop\quick_use"
print(base_dir)
input_dir = base_dir + "\**[a-z]"
print(input_dir)
print("#################################")
for file_dir in glob.glob(input_dir):
    print(file_dir)
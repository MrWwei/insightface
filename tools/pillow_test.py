import os
from PIL import Image

path = os.path.join(os.getcwd(),"/home/heisai/Pictures/output/00a317e1-e085-4ee8-8044-3bc01f32561b/00a317e1-e085-4ee8-8044-3bc01f32561b_00000.png")
img = Image.open(path)

size = img.size
print(size)# (3500, 3500)

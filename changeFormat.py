from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def view_pgm_file(file_path):
    try:
        img = Image.open(file_path)
        display(img)
    except Exception as e:
        print(f"Error: {e}")

def get_pgm_data(file_path):
    try:
        img = Image.open(file_path)
        img_array = np.array(img)
        # Perform further operations on the 'img_array' here
    except Exception as e:
        print(f"Error: {e}")

    return img_array

if 0:
    path1=r'D:\Downloads\faces\train\face\face00001.pgm'


    img_array = get_pgm_data(path1)

    plt.imshow(img_array,cmap='gray')

    plt.show()


from PIL import Image
import os

def convert_pgm_to_png(pgm_folder, png_folder):
    for filename in os.listdir(pgm_folder):
        if filename.lower().endswith('.pgm'):
            pgm_file = os.path.join(pgm_folder, filename)
            png_file = os.path.join(png_folder, os.path.splitext(filename)[0] + '.png')
            img = Image.open(pgm_file)
            img.save(png_file, format='PNG')

# Specify the folders for PGM and PNG images

pgm_folder = 'D:\\Downloads\\faces\\train\\face'
png_folder = 'D:\\Downloads\\faces\\Imagespng\\train\\face'

# Convert PGM images to PNG
convert_pgm_to_png(pgm_folder, png_folder)



pgm_folder ='D:\\Downloads\\faces\\train\\non-face'
png_folder = 'D:\\Downloads\\faces\\Imagespng\\train\\non-face'

# Convert PGM images to PNG
convert_pgm_to_png(pgm_folder, png_folder)


pgm_folder ='D:\\Downloads\\faces\\test\\face'
png_folder = 'D:\\Downloads\\faces\\Imagespng\\test\\face'

# Convert PGM images to PNG
convert_pgm_to_png(pgm_folder, png_folder)


pgm_folder ='D:\\Downloads\\faces\\test\\non-face'
png_folder = 'D:\\Downloads\\faces\\Imagespng\\test\\non-face'

# Convert PGM images to PNG
convert_pgm_to_png(pgm_folder, png_folder)




print('.')
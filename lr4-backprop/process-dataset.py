import zipfile
import io
import os
from pprint import pprint

import PIL
from PIL import Image
import colorama as clr

# rows = [
# 	["Hello", "🌎"],
# 	[],
# 	[None, ""]
# ]

def main():
    # for each folder in 2D_Geometric_Shapes_Dataset count the number of files
    side_size = 16
    zip_name = f"data/processed_{side_size}x{side_size}.zip"

    if os.path.exists(zip_name):
        response = input(f'{clr.Fore.YELLOW}File {
                         zip_name} already exists. Overwrite? [Y/n]{
                             clr.Style.RESET_ALL}')
        if response.lower() == 'y' or response.lower() == '':
            os.remove(zip_name)

    with zipfile.ZipFile("data/archive (1).zip", "r") as zi:
        in_filelist = zi.filelist
        files = []
        # count files in each folder 

        for f in in_filelist:
            folder = f.filename.split("/")[1]
            found = False
            for file in files:
                if file["folder"] == folder:
                    file["count"] += 1
                    found = True
                    break
            if not found:
                files.append({
                    "folder": folder,
                    "count": 1
                })
        pprint(files)

        # get folder names 
        folder_names = [file["folder"] for file in files]
        folder_names = sorted(folder_names)
        print(folder_names)

        encoded_names = []
        # for every name add corresponing number
        for i in range(len(folder_names)):
            encoded_names.append((folder_names[i], i))

        pprint(encoded_names)

        limit = 1000
        for f in in_filelist:
            if limit == 0:
                break
            # read data from the first image
            byte_data = zi.read(f) 
            img = Image.open(io.BytesIO(byte_data))


            # get pixel data from the image in rgba
            pixels = img.getdata()
            len_pixels = len(pixels)
            side_pixels = int(len_pixels ** 0.5)
            # print(f'{len_pixels}, {int(len_pixels ** 0.5)}x{int(len_pixels ** 0.5)}')
            img = img.convert("L")
            # resize the image to 32x32
            img = img.resize((side_size, side_size))

            pixel_bytes = img.tobytes()
            
            # if > 10 then convert to 0 else convert to 255
            pixel_bytes = bytearray(pixel_bytes)  # Convert to mutable bytearray
            for i in range(len(pixel_bytes)):
                if pixel_bytes[i] >= 0xf0:
                    pixel_bytes[i] = 0xff
                else:
                    pixel_bytes[i] = 0x00

            
            # create a new image from the pixel bytes
            img = Image.frombytes("L", img.size, bytes(pixel_bytes))  # Convert back to bytes

            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            png_data = buffer.getvalue()


            # add save whole png of the the image in data/processed.zip
            with zipfile.ZipFile(zip_name, "a") as zo:
                # filename='2D_Geometric_Shapes_Dataset/heptagon/heptagon_35107.png'
                # grab folder name of file (heptagon)
                folder = f.filename.split("/")[1]
                # grab number between _ and .png
                id = int(f.filename.split("/")[2].split("_")[1].split(".")[0])
                new_name = f"{encoded_names[folder_names.index(folder)][1]}_{id}.png"

                zo.writestr(new_name, png_data)
                print(f"{f.filename.split("/")[2]}({side_pixels}x{side_pixels}){clr.Fore.CYAN} -> {clr.Style.RESET_ALL}{new_name}({side_size}x{side_size})")
            limit -= 1

            # exit()




            # img.show()



        # convert the image to black and white



    
        


        # img.show()
            




if __name__ == '__main__':
    main()
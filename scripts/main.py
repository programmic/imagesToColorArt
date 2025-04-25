from PIL import Image
import random
import os
import json
from tqdm import tqdm
import shutil

def resizeImage(image: Image, max_width, max_height):
    original_width, original_height = image.size
    aspect_ratio = original_height / original_width

    # Korrigierter Faktor, da Terminal-Zeichen meist höher als breit sind
    terminal_aspect_ratio = 0.45  # Je nach Font evtl. anpassen

    new_width = min(max_width, original_width)
    new_height = int(new_width * aspect_ratio * terminal_aspect_ratio)

    if new_height > max_height:
        new_height = max_height
        new_width = int(new_height / (aspect_ratio * terminal_aspect_ratio))

    resized_image = image.resize((new_width, new_height))
    return resized_image

def create_colored_image(pSize: int, pC1: tuple, pC2: tuple) -> Image:
    # Create a new image with the specified size
    img_width = pSize * 2 + 1  # Two squares and a black line in between
    img_height = pSize
    new_image = Image.new("RGB", (img_width, img_height), "black")
    print(pC1, pC2)
    
    # Draw the first square
    for x in range(pSize):
        for y in range(pSize):
            new_image.putpixel((x, y), pC1)
    
    # Draw the second square
    for x in range(pSize + 1, img_width):
        for y in range(pSize):
            new_image.putpixel((x, y), pC2)
    
    return new_image

def findClosestColor(color, colors) -> tuple[str]:
    closest_color = None
    min_distance = float('inf')
    for c in colors:
        # Add a penalty for light colors to reduce their likelihood of being chosen
        distance = sum((color[i] - c[i]) ** 2 for i in range(3)) + (sum(c) / 3) ** 2
        if distance < min_distance:
            min_distance = distance
            closest_color = c
    return closest_color

def colorsToRGBList(colorsJSON):
    c: list = []
    for i in colorsJSON["colors"]:
        t= (i["rgb"][0], i["rgb"][1], i["rgb"][2])
        c.append(t)
    return c

def checkRandomPixel():
    pix = image.getpixel((random.randint(0, image.width - 1), random.randint(0, image.height - 1)))
    comp = create_colored_image(64,findClosestColor(pix, colorsRGB(colors)),pix)
    comp.show()

def transformImage(colorsRGB):
    newImage = Image.new("RGB", (image.width, image.height))
    for y in tqdm(range(image.height), desc="Transforming", ascii=True):
        print(f"\rProgress: {round(y / image.height * 100)}%", end="")
        for x in range(image.width):
            newImage.putpixel((x,y), findClosestColor(image.getpixel((x,y)),colorsRGB))
    return newImage

def givePixelByPixelBrightness(pixel) -> str:
    max_brightness = 255 * 3  # Maximum possible brightness
    blackThreshold = 85 * 3  # Threshold for reversing the gradient
    level = pixel[0] + pixel[1] + pixel[2]  # Sum of RGB values
    
    # Gradient string for brightness levels
    gradient = "@%#*+=-:. "
    gradient = "█▓▒@\$%&#ØØ¤¤◎oø*=+–•~¨°^`˝’’‘`·˙∘⠁⠀"
    #gradient = "@&%QWNM0gB#$DR8mHXKAUbGOpV4d9h6PkqwS2]ayjxY5Zoen[ul t13IfcFi|)7JvTLs?z/*cr!+><;=^,_:*'-.`"
    gradient = "█▓▒▒▒▒ "

    if level < blackThreshold:
        #max_brightness = blackThreshold
    
        gradient = gradient[::-1]
    # Calculate percentage and clamp it to [0, 1]
    percentage = max(0, min(level / max_brightness, 1))
    
    # Calculate the index within the gradient
    index = int(percentage * (len(gradient) - 1))
    
    return gradient[index]

def printIMageToconsole(image: Image, colors, mode: str = "rgb", watermark:str = "@programmic"):
    def printoutline():print(f"{code}{givePixelByPixelBrightness(image.getpixel((x, y)))}", end="", flush=False)

    print("\033c")
    newimage: Image = transformImage(colorsRGB)
    for y in range(image.height):
        for x in range(image.width):
            for i in colors["colors"]:
                if newimage.getpixel((x, y)) == tuple(i["rgb"]):
                    code = "\033[0;0m" + i["code"]
            if watermark != "":
                if (y == image.height -1 ) and (x >= image.width - len(watermark)):
                    print(watermark)
                    return
                else:
                    printoutline()
            else:
                printoutline()
                
        print("\033[0m")  # Reset color after each line
    return


if __name__ == '__main__':
    terminalSize = shutil.get_terminal_size()
    # Select a random image from the "images" directory
    images_dir = os.path.join(os.path.dirname(__file__), "../images")
    try:
        imgPath = os.path.join(images_dir, random.choice(os.listdir(images_dir)))
    except:
        print("\033[31mNo Image in folder\033[0;0m")
    image = resizeImage(Image.open(imgPath), terminalSize.columns, terminalSize.lines-4)

    # Load colors from colors.json
    with open('colors.json', 'r') as file:
        colors = json.load(file)
        for color in colors["colors"]:
            color["code"] = color["code"].encode().decode("unicode_escape")
    #checkRandomPixel()
    colorsRGB = colorsToRGBList(colors)
    
    printIMageToconsole(image, colors)


from PIL import Image, ImageFilter, ImageEnhance, ImageGrab
import os, math, random, json, shutil, colorsys, time
import helpful_functions as hf
import pygetwindow as gw
from tqdm import tqdm
import numpy as np

def timing(func):
    """
    A decorator to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper

def resizeImage(image: Image.Image, max_width, max_height):
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

@timing
def colorsToRGBList(colorsJSON):
    c: list = []
    for i in colorsJSON["colors"]:
        t= (i["rgb"][0], i["rgb"][1], i["rgb"][2])
        c.append(t)
    return c

def calculateLuminance(pixel: tuple[int, int, int]) -> float:
    """
    Calculate the luminance of an RGB pixel using the Rec. 709 formula.

    Args:
        pixel (tuple[int, int, int]): A tuple representing the RGB values of the pixel.

    Returns:
        float: The luminance value of the pixel.
    """
    r, g, b = pixel
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def sumColorDifference(c1: tuple | list, c2: tuple | list) -> int:
    return abs( calculateLuminance(c1) - calculateLuminance(c2) )

def posterizeImage(pImage: Image.Image, colors:dict, mode: str = "rgb") -> Image.Image:
    if mode not in ["rgb"]:
        raise ValueError(f"Error: {mode} is not a supported color mode for posterisation")
    newImage: Image.Image = Image.new("RGB", pImage.size)
    for x in tqdm(range(pImage.width),"Posterizing...", ascii=True):
        for y in range(pImage.height):
            closestColor = None
            closestColorDifference = math.inf
            for c in colors["colors"]:
                tmpSum = sum((pImage.getpixel((x, y))[i] - c[mode][i]) ** 2 for i in range(3))
                if  tmpSum < closestColorDifference:
                    closestColor = c
                    closestColorDifference = tmpSum
            newImage.putpixel((x,y), tuple(closestColor[mode]))
    return newImage

def giveCharacterByPixelBrightness(pixel, pGradient, gradientFilter: str="") -> str:
    max_brightness = 225*3   # Maximum possible brightness
    blackThreshold = 85 * 3  # Threshold for reversing the gradient
    level = sum(pixel) / 3  # Average of RGB values for normalization
    level = max(pixel)

    gradList = list(pGradient)
    if gradientFilter != "":
        gradList = [i for i in gradList if i in gradientFilter]
    pGradient = ''.join(gradList)

    for i in gradientFilter:
        if i not in pGradient:
            print(f"Element {i} not in gradient")

    if level < blackThreshold:
        max_brightness = blackThreshold
        pGradient = pGradient[::-1]
        pGradient = pGradient[:-1]

    # Calculate percentage and clamp it to [0, 1]
    percentage = max(0, min(level / max_brightness, 1))
    index = int(percentage * (len(pGradient) - 1))

    return pGradient[index]

def filterSobel(image, width, height, axis):
    kernelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] if axis == 1 else [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    gradient = [[0] * width for _ in range(height)]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            value = 0
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    value += image[x + kx, y + ky] * kernelX[ky + 1][kx + 1]
            gradient[y][x] = value
    return gradient

def filterSobel(image, width, height, axis):
    if axis == 1:
        kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    else:
        kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    imageData = [[image[x, y] if isinstance(image[x, y], int) else image[x, y][0] for x in range(width)] for y in range(height)]

    gradient = [[0] * width for _ in range(height)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            value = (
                imageData[y - 1][x - 1] * kernel[0][0] + imageData[y - 1][x] * kernel[0][1] + imageData[y - 1][x + 1] * kernel[0][2] +
                imageData[y][x - 1] * kernel[1][0] + imageData[y][x] * kernel[1][1] + imageData[y][x + 1] * kernel[1][2] +
                imageData[y + 1][x - 1] * kernel[2][0] + imageData[y + 1][x] * kernel[2][1] + imageData[y + 1][x + 1] * kernel[2][2]
            )
            gradient[y][x] = value

    return gradient

def filterSobelNew(image, width, height, axis):
    if isinstance(image[0, 0], int):
        img = np.array([[image[x, y] for x in range(width)] for y in range(height)], dtype=np.float32)
    else:
        img = np.array([[image[x, y][0] for x in range(width)] for y in range(height)], dtype=np.float32)

    if axis == 1:
        kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
    else:
        kernel = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32)

    gradient = np.zeros_like(img)

    gradient[1:-1,1:-1] = (
        img[0:-2,0:-2] * kernel[0,0] + img[0:-2,1:-1] * kernel[0,1] + img[0:-2,2:] * kernel[0,2] +
        img[1:-1,0:-2] * kernel[1,0] + img[1:-1,1:-1] * kernel[1,1] + img[1:-1,2:] * kernel[1,2] +
        img[2:,0:-2] * kernel[2,0] + img[2:,1:-1] * kernel[2,1] + img[2:,2:] * kernel[2,2]
    )

    return gradient.tolist()

@timing
def getEdgeDirection(image) -> Image.Image:
    width, height = image.size
    imageData = image.load()

    dx = filterSobelNew(imageData, width, height, axis=1)
    dy = filterSobelNew(imageData, width, height, axis=0)

    dx = np.array(dx)
    dy = np.array(dy)

    direction = np.arctan2(dy, dx)

    directionDeg = np.degrees(direction)
    directionDeg = (directionDeg + 360) % 360
    directionNormalized = directionDeg / 360

    hsv = np.stack((directionNormalized, np.ones_like(directionNormalized), np.ones_like(directionNormalized)), axis=-1)
    hsv_flat = hsv.reshape(-1, 3)

    rgb_flat = []
    for h in tqdm(hsv_flat, desc="Converting HSV to RGB", ascii=True):
        rgb_flat.append(colorsys.hsv_to_rgb(*h))
    rgb_flat = np.array(rgb_flat)

    rgb = (rgb_flat * 255).astype(np.uint8).reshape((height, width, 3))

    output = Image.fromarray(rgb, mode='RGB')

    return output

def downsampleGrayscale(pImage: Image.Image, levels: int = 1):
    levels: int = 3
    width = pImage.width // levels
    height = pImage.height // levels

    dsImage = Image.new("L", (width, height))
    for x in range(width):
        for y in range(height):
            pixel_sum = 0
            samples = 0
            for ix in range(levels):
                for iy in range(levels):
                    ox = x * levels + ix
                    oy = y * levels + iy
                    pixel_sum += pImage.getpixel((ox, oy))
                    samples += 1
            avg = int(pixel_sum / samples)
            dsImage.putpixel((x, y), avg)
    return dsImage

def printDebugLevels(pImage, colorTable):
    values: dict = {}
    for x in range(pImage.width):
        for y in range(pImage.height):
            pixel = pImage.getpixel((x, y))
            if pixel in values:
                values[pixel] += 1
            else:
                values[pixel] = 1
    
    max_brightness = 225*3   # Maximum possible brightness
    blackThreshold = 85 * 3  # Threshold for reversing the gradient
    for pixel in values.keys():
        
        level = sum(pixel) / 3  # Average of RGB values for normalization
        percentage = max(0, min(level / max_brightness, 1))
        if level < blackThreshold:
            max_brightness = blackThreshold

        colorCode = "UNDEFINED"

        for i in  colorTable["colors"]:
            if tuple(i["rgb"]) == pixel:
                colorCode = i["code"]
                colorName = i["name"]
        perc = round(values[pixel] / (pImage.width * pImage.height) * 100, 2)
        print("p:", hf.lenformat(pixel,15), "  ||  %:", hf.lenformat(str(round(percentage*100, 2)), 5), "  ||  l:",hf.lenformat(str(round(level)), 3),"  ||  &:", hf.lenformat(str(round(level/max_brightness*100)),3),f"{colorCode}[#]\033[0;0m", hf.lenformat(colorName, 15), hf.lenformat(values[pixel],5), f"{perc}%")

def printImageToconsole(
        image: Image.Image,
        brightness: Image.Image,
        colors,
        outlines: Image.Image = None,
        outlineThreshold: int = 60,
        mode: str = "rgb",
        watermark: str = "@programmic",
        pGrad: str = "Xx.",
        grad: str = "||\\--/||\\--/||"
    ):
    """
    Prints the image to the console with optional outlines and a watermark.

    :param image: The color image to render.
    :param colors: The color palette in JSON format.
    :param outlines: The outline image to overlay (optional).
    :param outlineThreshold: The brightness threshold for outlines.
    :param mode: The mode for rendering (default is "rgb").
    :param watermark: The watermark to display at the bottom-right corner.
    :param grad: The gradient string for outlines.
    """
    if not 0 < outlineThreshold <= 255:
        raise ValueError(f"{outlineThreshold} is not in the range 1-255")
    if outlines and outlines.size != image.size:
        raise ValueError(f"Image size {image.size} is unequal to outline size {outlines.size}")

    def printoutline(x, y, pGrad):
        """
        Prints the outline or the color pixel based on the outline threshold.
        """
        if outlines:
            pixel = outlines.getpixel((x, y))
            value = (pixel[0] + pixel[1] + pixel[2]) / 3
            if value >= outlineThreshold:
                print(f"\033[97;0m{hueToChar(pixel, grad)}", end="", flush=True)
            else:
                simpleout(x, y, pGrad)
        else:
            simpleout(x, y, pGrad)

    def simpleout(x, y, pGrad):
        """
        Prints the color pixel with brightness.
        """
        pixelRGB = image.getpixel((x, y))  # Get the pixel's RGB value
        pixelBrightness = brightness.getpixel((x, y))  # Get the same pixel for brightness calculation
    
        # Match the pixel's RGB value with the palette
        for color in colors["colors"]:
            #print(pixel_rgb, tuple(color["rgb"]), pixel_rgb == tuple(color["rgb"]))
            if pixelRGB == tuple(color["rgb"]):  # Ensure both are tuples of integers
                color_code = color["code"]
                break
        else:
            raise ValueError(f"could not find matching code for color {pixelRGB}")
    
        # Print the pixel with the corresponding color and brightness
        print(f"{color_code}{giveCharacterByPixelBrightness(pixelBrightness, pGrad, gradientFilter="")}", end="", flush=False)

    for y in range(image.height):
        for x in range(image.width):
            if watermark and (y == image.height - 1) and (x >= image.width - len(watermark)):
                print(watermark, end="")
                break
            printoutline(x, y, pGrad)
        print("\033[0m")  # Reset color after each line
    return


@timing
def adjustEdgeDetectionDetail(image: Image.Image, blur_radius: int = 2, contrast_factor: float = 0.75) -> Image.Image:
    """
    Adjusts the detail level of edge detection by applying a Gaussian blur and adjusting contrast.

    :param image: The input grayscale image.
    :param blur_radius: The radius for the Gaussian blur. Higher values reduce detail.
    :param contrast_factor: The factor to adjust contrast. Lower values reduce detail.
    :return: The processed image with reduced detail.
    """
    # Apply Gaussian blur to smooth out fine details
    blurred_image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Adjust contrast to further reduce detail
    enhancer = ImageEnhance.Contrast(blurred_image)
    adjusted_image = enhancer.enhance(contrast_factor)
    
    return adjusted_image

@timing
def brightnessMask(imgColor: Image.Image, imgValue: Image.Image) -> Image.Image:
    maskedImage: Image.Image = Image.new("RGB", imgColor.size)
    maskedPixels = maskedImage.load()

    for x in range(imgColor.width):
        for y in range(imgColor.height):
            pixelC = imgColor.getpixel((x, y))
            pixelV = imgValue.getpixel((x, y))
            r = pixelC[0] * pixelV / 255
            g = pixelC[1] * pixelV / 255
            b = pixelC[2] * pixelV / 255
            maskedPixels[x, y] = (int(r), int(g), int(b))
    return maskedImage

@timing
def brightnessMaskNew(imgColor: Image.Image, imgValue: Image.Image) -> Image.Image:
    color_array = np.array(imgColor, dtype=np.float32)
    value_array = np.array(imgValue.convert('L'), dtype=np.float32)
    value_array = value_array[:, :, np.newaxis]
    masked_array = color_array * (value_array / 255.0)
    masked_array = masked_array.clip(0, 255).astype(np.uint8)
    maskedImage = Image.fromarray(masked_array, mode="RGB")

    return maskedImage

def hueToChar(rgb, gradient_string):
    """
    Converts an RGB color to a hue and selects a character from a gradient string.

    Args:
        rgb (tuple[int, int, int]): Red, Green, Blue values (0-255)
        gradient_string (str): A string of characters representing the gradient

    Returns:
        str: One character from the gradient string based on hue
    """
    r, g, b = rgb
    # Normalize RGB to 0–1
    normR, normG, normB = r / 255, g / 255, b / 255

    # Convert to HSV (Hue in [0.0, 1.0])
    hue, _, _ = colorsys.rgb_to_hsv(normR, normG, normB)

    # Map hue to index in gradient
    index = int(hue * (len(gradient_string) - 1))
    return gradient_string[index]


if __name__ == '__main__':
    # Load settings data from JSON
    with open('settings.json', encoding='utf-8') as f:
        settings = json.load(f)["settings"]
    

    print("\033c")
    for i in settings:
        print(hf.lenformat(i,25), settings[i])

    print()
    print(str(settings["load"]))
    print(list(settings["loadTypes"]))
    

    if settings["load"] not in settings["loadTypes"]:
        print(f"Error: {settings["load"]} not in {settings["loadTypes"]}")
        exit(1)
    
    if settings["gradientType"] not in settings["gradientTypes"]:
        print(f"Error: {settings["gradientType"]} not in {settings["gradientTypes"]}")
        exit(1)
    
    s_drawOutline = settings["outline"]
    s_gradient = settings["gradients"]["minimalist"]
    s_outlineStrength = settings["outlineStrength"]
    s_outlineBlurRadius = settings["outlineBlurRadius"]
    s_outlineContrastFactor = settings["outlineContrastFactor"]
    s_watermark = settings["watermark"]
    s_exportImage = settings["exportImage"]




    images_dir = os.path.join(os.path.dirname(__file__), "../images")
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
    if not images:
        print("\033[31mNo images found in the folder\033[0;0m")
        exit(1)

    if settings["load"] == "select":
        for i in range(len(images)):
            print(hf.lenformat(i, 2, " ", "front") + ". " + images[i])
        validInput = False
        while not validInput:
            selImage = input(f"Select image... ( Input Number between 0 and {len(images)-1} ):  ")
            if selImage.isnumeric():
                if  0 <= int(selImage) <= len(images)-1:
                    validInput = True
                    break
            print(f"[ {selImage} ] is not a acceptable input. Please input a Integer between 0 and {len(images)-1}.  ")
        imgPath = os.path.join(images_dir, images[int(selImage)])
    elif settings["load"] == "random":
        imgPath = os.path.join(images_dir, random.choice(images))
    else:
        print("Unsupported image selection mode... Exiting")
        exit(1)
    image = Image.open(imgPath)
    

    if settings["setSettingsInTerminal"]:

        validInput = False
        while not validInput:
            sel = input("Please select wether or not to draw a outline ( Y/N ):  ")
            if sel.lower() == "y" or sel == "1" or sel == "1":
                s_drawOutline = True
                validInput = True
            elif sel.lower() == "n" or sel == "0":
                s_drawOutline = False
                validInput = True
            else:
                print(f"[ {sel} ] is not a acceptable input. Draw outline? ( Y/N ):  ")

        validInput = False
        while not validInput:
            for i in range(len(settings["gradientTypes"])):
                print(hf.lenformat(i, 2, " ", "front") + ". " + settings["gradientTypes"][i])
            sel = input(f"Please select shading gradiant ({0} - {len(settings["gradientTypes"])-1}):  ")
            if sel.isnumeric():
                if  0 <= int(sel) <= len(settings["gradientTypes"])-1:
                    validInput = True
                    if settings["gradientTypes"][int(sel)] != "random":
                        gradType = settings["gradientTypes"][int(sel)]
                    else:
                        gradType = settings["gradientTypes"][random.randint(0, len(settings["gradients"])-1)]
                    s_gradient = settings["gradients"][gradType]
                    print("Selected gradient:", s_gradient)

        validInput = False
        while not validInput:
            sel = input("Please select wether or not to draw a watermark ( Y/N/C ):  ")
            if sel.lower() == "y" or sel == "1":
                s_watermark = settings["watermark"]
                validInput = True
            elif sel.lower() == "n" or sel == "0":
                s_watermark = ""
                validInput = True
            elif sel.lower == "c":
                s_watermark = input("Please input watermark:  ( <any> )  ")
                validInput
            else:
                print(f"[ {sel} ] is not a acceptable input. Draw watermark? Enter 'C' to enter custom watermark ( Y/N/C ):  ")
                validInput = True
    
            

        validInput = False
        while not validInput:
            sel = input("Please select wether to export Image ( Y/N ):  ")
            if sel.lower() == "y" or sel == "1":
                s_exportImage = True
                validInput = True
            elif sel.lower() == "n" or sel == "0":
                s_exportImage = False
                validInput = True
            else:
                print(f"[ {sel} ] is not a acceptable input. Export Image? ( Y/N ):  ")


    # Load colors from colors.json
    with open('colors.json', 'r') as file:
        colorsJSON = json.load(file)
        colorsRGB = colorsToRGBList(colorsJSON)  # Convert to a list of RGB tuples
        for color in colorsJSON["colors"]:
            color["code"] = color["code"].encode().decode("unicode_escape")


    terminalSize = shutil.get_terminal_size()
    
    print("\n\n")
    
    # Calculate the new dimensions for resizing
    max_width = terminalSize.columns
    max_height = terminalSize.lines - 4
    new_width, new_height = resizeImage(image, max_width, max_height).size

    if s_drawOutline:
        print("Processing Edge Data...",end="")
        grayscaleImage = image.convert("L")
        edgeDetect = adjustEdgeDetectionDetail(grayscaleImage, s_outlineBlurRadius, s_outlineContrastFactor).filter(ImageFilter.FIND_EDGES)
        edgeDirections = getEdgeDirection(edgeDetect)
        edgeDirectionMask = brightnessMaskNew(edgeDirections, edgeDetect)
        printOutline = edgeDirectionMask.resize((new_width, new_height))



    # Resize both the color image and the outline image to the same dimensions
    colorImage = posterizeImage(image.resize((new_width, new_height)), colorsJSON)
    brightnessImage = image.resize((new_width, new_height))
    

    if s_drawOutline:
        printImageToconsole(colorImage, 
                            brightnessImage,
                            colorsJSON,
                            printOutline,
                            s_outlineStrength,
                            pGrad=s_gradient,
                            watermark=s_watermark)
    else:
        printImageToconsole(colorImage,
                            brightnessImage,
                            colorsJSON,
                            pGrad=s_gradient,
                            watermark=s_watermark)
    if s_exportImage:
        s_exportLocation = os.getcwd() + settings["exportLocation"]
        print(s_exportLocation)
        windows = gw.getAllTitles()
        terminalStr = next((window for window in windows if "cmd.exe" in window), None)
        if terminalStr == None:
            print("No terminal found. Perhaps programm is running in IDE-integrated terminal?\nExiting...")
            exit(1)
        terminal: gw.Win32Window = gw.getWindowsWithTitle(terminalStr)[0]
        terminalDimensions = (terminal.width, terminal.height)
        terminalPosition = (terminal.topleft[0], terminal.topright[1])
        print(terminalDimensions, terminalPosition)
        time.sleep(1)
        screenshot = ImageGrab.grab(bbox=(terminalPosition[0], terminalPosition[1], terminalPosition[0]+terminalDimensions[0], terminalPosition[1]+terminalDimensions[1]))
        # Ensure the export directory exists
        if not os.path.exists(os.path.dirname(s_exportLocation)):
            os.makedirs(os.path.dirname(s_exportLocation))
        exportPath: os.PathLike = s_exportLocation + str(hf.timeFormat(time.time())) + "_" + imgPath.split("\\")[-1].split(".")[0] + ".jpg"
        print("Saving image as", exportPath)
        screenshot.save(exportPath)
    if settings["printDebugInformation"]: printDebugLevels(colorImage, colorsJSON)
# Images to Color Art

## Description
This project transforms images into ASCII-style art using terminal colors. It maps the colors of an image to a predefined palette and displays the result in the terminal. The program uses brightness gradients and color interpolation to create visually appealing representations of images.

## Features
- Converts images into terminal-friendly color art.
- Supports a customizable color palette defined in `colors.json`.
- Uses brightness gradients for detailed rendering.
- Displays the transformed image directly in the terminal.

## How It Works
1. The program resizes the input image to fit the terminal dimensions.
2. Each pixel is mapped to the closest color in the palette defined in `colors.json`.
3. Brightness levels are calculated for each pixel, and a gradient character is selected to represent the pixel's intensity.
4. The result is printed to the terminal using ANSI escape codes for colors.

## Setup Instructions
1. Clone the repository or download the project files.
2. Ensure you have Python installed on your system.
3. Set up a virtual environment:
   - On Windows: `python -m venv .venv`
   - On macOS/Linux: `python3 -m venv .venv`
4. Activate the virtual environment:
   - On Windows: `.\.venv\Scripts\activate`
   - On macOS/Linux: `source .venv/bin/activate`
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
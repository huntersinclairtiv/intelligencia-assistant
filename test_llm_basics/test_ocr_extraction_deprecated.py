from PIL import Image
import pytesseract

# Open an image file
image_path = 'figures/figure-1-1.jpg'
img = Image.open(image_path)

# Use Tesseract OCR to extract text
text = pytesseract.image_to_string(img)

with open('output.txt', 'w') as f:
    f.write(text)

# Preprocess the text as needed
# Tokenization and further processing can be done here

# Use the processed text as input to a GPT model

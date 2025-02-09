import easyocr

# Path to the input image
image_path = '/path/to/your/image.png'

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Use EasyOCR to extract text
extracted_text_easyocr = reader.readtext(image_path, detail=0)

# Print the extracted text
print(' '.join(extracted_text_easyocr))

print("a new line")

print("edited in local")
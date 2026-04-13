import os
from rembg import remove, new_session
from PIL import Image
import io

session = new_session("u2net")
# Use a file that definitely exists
test_file = 'mango/Anthracnose/20211008_124249 (Custom).jpg'

with open(test_file, 'rb') as i:
    input_data = i.read()
output_data = remove(input_data, session=session)
img = Image.open(io.BytesIO(output_data))
img.save("test_masked.png")
print(f"Saved test_masked.png for {test_file}")

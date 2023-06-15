from PIL import Image, ImageOps

# Get the screen.png image
image = Image.open('screen.png')

# Resize the image to 100x100
image = image.resize((500, 500))

# Draw a black 1 pixel border around the image
image = ImageOps.expand(image, border=2, fill='black')

# Flip the image horizontally
image = image.transpose(Image.FLIP_LEFT_RIGHT)

# Save the resized image
image.save('screen_resized.png')
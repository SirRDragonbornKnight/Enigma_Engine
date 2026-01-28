# PWA Icons Placeholder

The ForgeAI PWA requires icon images in multiple sizes. These should be placed in this directory:

- icon-72.png (72x72)
- icon-96.png (96x96)
- icon-128.png (128x128)
- icon-144.png (144x144)
- icon-152.png (152x152)
- icon-192.png (192x192) - Used for Android
- icon-384.png (384x384)
- icon-512.png (512x512) - Used for splash screen

You can generate these icons from any image using online tools like:
- https://realfavicongenerator.net/
- https://www.pwabuilder.com/imageGenerator

Or use the Python Pillow library:
```python
from PIL import Image, ImageDraw, ImageFont

# Create a simple icon
img = Image.new('RGB', (512, 512), color='#1a1a2e')
draw = ImageDraw.Draw(img)

# Draw ForgeAI logo/text
# (Add your custom design here)

# Save in multiple sizes
sizes = [72, 96, 128, 144, 152, 192, 384, 512]
for size in sizes:
    resized = img.resize((size, size), Image.LANCZOS)
    resized.save(f'icon-{size}.png')
```

For now, the PWA will work without icons, but they are recommended for a better user experience.

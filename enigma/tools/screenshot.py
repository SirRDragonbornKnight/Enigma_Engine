"""
Screenshot helper. For cross-platform use PIL.ImageGrab on supported platforms.
"""
try:
    from PIL import ImageGrab
except Exception:
    ImageGrab = None

def take_screenshot(path="screenshot.png"):
    if ImageGrab is None:
        raise RuntimeError("Pillow not installed or ImageGrab not available on this platform")
    img = ImageGrab.grab()
    img.save(path)
    return path

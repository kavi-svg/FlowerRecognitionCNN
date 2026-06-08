"""Generate simple test flower images for demonstration."""
import pathlib
from PIL import Image, ImageDraw

def create_test_images():
    """Create 3 simple test images in downloads folder."""
    output_dir = pathlib.Path.home() / "Downloads"
    
    # Image 1: Red flower (class1)
    img1 = Image.new("RGB", (200, 200), color="white")
    draw1 = ImageDraw.Draw(img1)
    draw1.ellipse([50, 50, 150, 150], fill="red", outline="darkred")
    draw1.ellipse([70, 70, 130, 130], fill="yellow")
    img1.save(output_dir / "flower1.jpg")
    print(f"✓ Created: {output_dir / 'flower1.jpg'}")
    
    # Image 2: Blue flower (class2)
    img2 = Image.new("RGB", (200, 200), color="white")
    draw2 = ImageDraw.Draw(img2)
    draw2.ellipse([50, 50, 150, 150], fill="blue", outline="darkblue")
    draw2.ellipse([70, 70, 130, 130], fill="yellow")
    img2.save(output_dir / "flower2.jpg")
    print(f"✓ Created: {output_dir / 'flower2.jpg'}")
    
    # Image 3: Yellow flower (class1)
    img3 = Image.new("RGB", (200, 200), color="white")
    draw3 = ImageDraw.Draw(img3)
    draw3.ellipse([50, 50, 150, 150], fill="gold", outline="orange")
    draw3.ellipse([70, 70, 130, 130], fill="orange")
    img3.save(output_dir / "flower3.jpg")
    print(f"✓ Created: {output_dir / 'flower3.jpg'}")
    
    print(f"\nAll test images created in: {output_dir}")

if __name__ == "__main__":
    create_test_images()

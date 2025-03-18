import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def enhance_skin_texture(image_path, output_path, texture_strength=0.3, noise_scale=0.02, detail_enhancement=1.5):
    """
    Enhances skin texture of a generated face image to reduce the "plastic" appearance.

    Args:
        image_path (str): Path to the input image
        output_path (str): Path to save the enhanced image
        texture_strength (float): Controls the strength of texture overlay (0.0-1.0)
        noise_scale (float): Scale of the noise pattern (higher = finer grain)
        detail_enhancement (float): Controls the level of detail enhancement
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert to float for processing
    img_float = img.astype(np.float32) / 255.0

    # Step 1: Create a skin mask (simplified approach)
    # Convert to YCrCb color space for better skin detection
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Define skin color range in YCrCb
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)

    # Create skin mask
    skin_mask = cv2.inRange(img_ycrcb, lower_skin, upper_skin)

    # Add some refinement to the skin mask
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    _, skin_mask = cv2.threshold(skin_mask, 127, 255, 0)
    skin_mask = cv2.dilate(skin_mask, None, iterations=2)
    skin_mask = cv2.erode(skin_mask, None, iterations=1)
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

    # Convert to float [0,1]
    skin_mask_float = skin_mask.astype(np.float32) / 255.0
    skin_mask_float = np.stack([skin_mask_float, skin_mask_float, skin_mask_float], axis=2)

    # Step 2: Generate realistic skin texture
    # Create noise texture
    h, w = img.shape[:2]
    noise = np.random.normal(0, 1, (h, w, 3)).astype(np.float32)

    # Scale the noise
    noise = noise * noise_scale

    # Apply Gaussian filtering to make the noise more natural
    for c in range(3):
        noise[:, :, c] = gaussian_filter(noise[:, :, c], sigma=1.0)

    # Step 3: Extract and enhance details
    # Apply bilateral filter to separate details from base
    img_base = cv2.bilateralFilter(img, 9, 75, 75).astype(np.float32) / 255.0
    img_detail = img_float - img_base

    # Enhance details
    img_detail_enhanced = img_detail * detail_enhancement

    # Step 4: Combine everything
    # Apply noise to skin areas
    enhanced_img = img_base + img_detail_enhanced + (noise * skin_mask_float * texture_strength)

    # Ensure values are in valid range
    enhanced_img = np.clip(enhanced_img, 0.0, 1.0)

    # Convert back to uint8
    enhanced_img = (enhanced_img * 255).astype(np.uint8)

    # Save the result
    cv2.imwrite(output_path, enhanced_img)
    print(f"Enhanced image saved to {output_path}")

    return enhanced_img



if __name__ == "__main__":
    enhance_skin_texture("input_face.jpg", "enhanced_face.jpg", texture_strength=0.4, noise_scale=0.03, detail_enhancement=1.8)

import cv2
import numpy as np

# Here are some utility functions for image processing tasks.
# Is like a library of common image processing operations used across the application.
# Use utils folder to store such utility scripts that provide reusable functions like these.

def get_part_silhouette_filled(roi: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Extracts the silhouette from the given image using edge detection and fills it with the original pixel values.

    Args:
        roi (np.ndarray): Input image region of interest in BGR format or already in grayscale.
        target_size (tuple): Desired output size (width, height).

    Returns:
        np.ndarray: Image with the silhouette filled with original pixel values.
    """
    img_uint8 = img_to_uint8(roi)  # Convert the input image to uint8 format if necessary.
    img_gray = img_to_gray(img_uint8)  # Convert the input image to grayscale if it is in BGR format.

    silhouette = get_otsu_silhouette(img_gray)  # Generate a binary silhouette using Otsu's thresholding method.
    
    fill_silhouette = fill_silhouette_with_original(silhouette, img_uint8)  # Fill the silhouette with original pixel values.
    fill_silhouette_clahe = apply_clahe_to_img_uint8(fill_silhouette)  # Apply CLAHE to enhance contrast of the filled silhouette.

    resized = cv2.resize(fill_silhouette_clahe, target_size)  # Resize to target size.

    return np.expand_dims(img_to_float32(resized), axis=-1)  # Convert to float32, normalize to [0, 1], and add channel dimension.

# == Basic image manipulation functions: masking, motion detection, ROI extraction ==
def mask_gap(roi: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Masks a rectangular gap in the image by setting the specified region to zero.

    Args:
        roi (np.ndarray): Input image region of interest.
        x (int): X-coordinate of the top-left corner of the rectangle.
        y (int): Y-coordinate of the top-left corner of the rectangle.
        width (int): Width of the rectangle.
        height (int): Height of the rectangle.

    Returns:
        np.ndarray: Image with the specified region masked.
    """
    y1, x1 = y+height, x+width  # Calculate bottom-right corner coordinates.
    roi[y:y1, x:x1] = 0  # Set the specified rectangle area to zero.
    return roi  # Return the modified image.

def detect_motion(current_frame: np.ndarray, previous_frame: np.ndarray, threshold: int = 80) -> np.ndarray:
    """
    Detects motion between two frames by computing the absolute difference.

    Args:
        current_frame (np.ndarray): The current frame in BGR format.
        previous_frame (np.ndarray): The previous frame in BGR format.
        threshold (int): Threshold value to binarize the motion mask.

    Returns:
        np.ndarray: Binary image representing areas of motion.
    """
    if current_frame is None or previous_frame is None:
        return False    # Return False if either frame is None.

    diff = cv2.absdiff(current_frame, previous_frame)  # Compute absolute difference between frames.
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # Convert difference image to grayscale.
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)  # Binarize the grayscale image.

    return np.sum(mask) > 0  # Return True if there is any motion detected, else False.

def get_roi(frame: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Extracts a region of interest (ROI) from the given frame.

    Args:
        frame (np.ndarray): Input image frame.
        x (int): X-coordinate of the top-left corner of the ROI.
        y (int): Y-coordinate of the top-left corner of the ROI.
        width (int): Width of the ROI.
        height (int): Height of the ROI.

    Returns:
        np.ndarray: Extracted region of interest.
    """
    return frame[y:y+height, x:x+width]  # Return the specified ROI from the frame.

# == Images to different formats: uint8, float32 ==
def img_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Converts an image to uint8 format, scaling if necessary.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Image in uint8 format.
    """
    if image.dtype == np.uint8:
        return image  # Return as is if already in uint8 format.
    elif image.max() <= 1.0:
        return (image * 255).astype(np.uint8)  # Scale to 0-255 if values are in [0, 1].
    else:
        return image.astype(np.uint8)  # Convert to uint8 without scaling if values are already in a larger range.
    
def img_to_float32(image: np.ndarray) -> np.ndarray:
    """
    Converts an image to float32 format, scaling if necessary.

    Args:
        image (np.ndarray): Input image.
    Returns:
        np.ndarray: Image in float32 format with values in [0, 1].
    """
    if image.dtype == np.float32:
        return image  # Return as is if already in float32 format.
    elif image.max() > 1.0:
        return (image / 255.0).astype(np.float32)  # Scale to [0, 1] if values are in a larger range.
    else:
        return image.astype(np.float32)  # Convert to float32 without scaling if values are already in [0, 1].

# == Image processing functions: to gray, clahe img uint8 == 
def img_to_gray(image: np.ndarray) -> np.ndarray:
    """
    Converts a BGR image to grayscale or returns the image as is if it's already grayscale.

    Args:
        image (np.ndarray): Input image in BGR format or already in grayscale.

    Returns:
        np.ndarray: Grayscale image.
    """
    if image.ndim == 3 and image.shape[-1] == 3:  # Check if the image is in BGR format.
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the input image from BGR to grayscale.
    elif image.ndim == 2:  # If the image is already grayscale, return it as is.
        return image    # Return the input image as is if it is already in grayscale format.
    
def apply_clahe_to_img_uint8(image_uint8: np.ndarray) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the contrast of the image while preventing over-amplification of noise. 
    The function handles both grayscale and BGR images.

    Args:
        image_uint8 (np.ndarray): Input image in grayscale format (single channel) or BGR format (three channels) in uint8 data type.

    Returns:
        np.ndarray: Image with enhanced contrast after applying CLAHE. 
        The output image will have the same number of channels as the input image (grayscale or BGR) and will be in uint8 data type.
    """

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create a CLAHE object with specified parameters.

    if image_uint8.ndim == 3 and image_uint8.shape[-1] == 3:  # Check if the image is in BGR format.
        channels = cv2.split(image_uint8)  # Split the BGR image into its individual channels.
        channels_clahe = [clahe.apply(channel) for channel in channels]  # Apply CLAHE to each channel separately.
        clahe_out = cv2.merge(channels_clahe)  # Merge the CLAHE-enhanced channels back into a BGR image.
    elif image_uint8.ndim == 3 and image_uint8.shape[-1] == 1:  # Check if the image is in grayscale format with a single channel.
        channels_clahe = clahe.apply(np.squeeze(image_uint8))  # Apply CLAHE to the single channel image.
        clahe_out = np.expand_dims(channels_clahe, axis=-1)  # Add a channel dimension back to the CLAHE-enhanced image.
    else:
        clahe_out = clahe.apply(image_uint8)  # Apply CLAHE directly if the image is already in a compatible format.

    return clahe_out  # Return the CLAHE-enhanced image.

# == Silhouette generation and filling functions ==
def get_otsu_silhouette(img_gray: np.ndarray) -> np.ndarray:
    """
    Generates a binary silhouette using Otsu's thresholding method.

    Args:
        img_gray (np.ndarray): Input image in grayscale format.

    Returns:
        np.ndarray: Binary silhouette image.
    """
    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # Create binary mask using Otsu's thresholding method.
    binary_blurred = cv2.GaussianBlur(mask, (3, 3), 0)  # Apply Gaussian blur to the mask.
    return binary_blurred  # Return the blurred binary silhouette.

def binary_to_3channels(image: np.ndarray) -> np.ndarray:
    """
    Converts a grayscale image to BGR format or returns the image as is if it's already in BGR format.

    Args:
        image (np.ndarray): Input image in grayscale format or already in BGR format.

    Returns:
        np.ndarray: Image in BGR format.
    """
    if image.ndim == 2:  # Check if the image is in grayscale format.
        return np.stack((image,)*3, axis=-1)  # Stack the grayscale image into three channels to create a BGR image.
    elif image.ndim == 3 and image.shape[-1] == 3:  # If the image is already in BGR format, return it as is.
        return image    # Return the input image as is if it is already in BGR format.
    
def fill_silhouette_with_original(silhouette: np.ndarray, original: np.ndarray) -> np.ndarray:
    """
    Fills the silhouette image with the original pixel values where the silhouette is white.
    This is the default way to fill the silhouette, where the background is white.

    Args:
        silhouette (np.ndarray): Binary silhouette image.
        original (np.ndarray): Original image in BGR format.

    Returns:
        np.ndarray: Image where the silhouette is filled with original pixel values and background is white.
    """
    bgr_silhouette = binary_to_3channels(silhouette)  # Convert the binary silhouette to BGR format.
    filled = np.where(bgr_silhouette == 255, original, 255)  # Fill the silhouette with original pixel values where the mask is white and set others to white.

    return filled  # Return the filled silhouette image.

def fill_silhouette_with_original_background_inverted(silhouette: np.ndarray, original: np.ndarray) -> np.ndarray:
    """
    Fills the silhouette image with the original pixel values where the silhouette is white.
    This is the inverted way to fill the silhouette, where the background is black.

    Args:
        silhouette (np.ndarray): Binary silhouette image.
        original (np.ndarray): Original image in BGR format.

    Returns:
        np.ndarray: Image where the silhouette is filled with original pixel values and background is black.
    """
    bgr_silhouette = binary_to_3channels(silhouette)  # Convert the binary silhouette to BGR format.
    filled = np.where(bgr_silhouette == 255, original, 0)  # Fill the silhouette with original pixel values where the mask is white and set others to black.

    return filled  # Return the filled silhouette image.


# == Inference preprocessing pipeline tools ==
# These functions are used by SequenceSettings to apply a JSON-defined preprocessing
# pipeline to frames before inference. Each function corresponds to one pipeline tool.

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to each channel of an RGB (or BGR) image independently.

    Enhances local contrast without amplifying noise, matching the preprocessing
    applied during model training (training_pytorch_float.py: apply_clahe_rgb).

    Args:
        image (np.ndarray): Input image in RGB or BGR uint8 format, shape (H, W, 3).

    Returns:
        np.ndarray: CLAHE-enhanced image with the same shape and dtype as the input.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(image)
    return cv2.merge([clahe.apply(ch) for ch in channels])


def apply_roi_crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Crop a rectangular region of interest from an image.

    Args:
        image (np.ndarray): Input image.
        x (int): X-coordinate of the top-left corner.
        y (int): Y-coordinate of the top-left corner.
        w (int): Width of the region in pixels.
        h (int): Height of the region in pixels.

    Returns:
        np.ndarray: Cropped image of shape (h, w, C).
    """
    return image[y:y + h, x:x + w]


def put_black_circle(image: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray:
    """
    Draw a filled black circle on the image to mask an irrelevant area (e.g. screw hole,
    reflective bolt) before inference.

    Args:
        image (np.ndarray): Input image.
        cx (int): X-coordinate of the circle center.
        cy (int): Y-coordinate of the circle center.
        radius (int): Radius of the circle in pixels. If 0, the image is returned unchanged.

    Returns:
        np.ndarray: Copy of the image with the circle drawn.
    """
    result = image.copy()
    if radius > 0:
        cv2.circle(result, (cx, cy), radius, (0, 0, 0), thickness=-1)
    return result


def put_black_rectangle(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Draw a filled black rectangle on the image to mask an irrelevant area before inference.

    Args:
        image (np.ndarray): Input image.
        x (int): X-coordinate of the top-left corner.
        y (int): Y-coordinate of the top-left corner.
        width (int): Width of the rectangle in pixels. If 0, the image is returned unchanged.
        height (int): Height of the rectangle in pixels. If 0, the image is returned unchanged.

    Returns:
        np.ndarray: Copy of the image with the rectangle drawn.
    """
    result = image.copy()
    if width > 0 and height > 0:
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 0, 0), thickness=-1)
    return result


def resize_to_training_resolution(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize the image to the resolution used during model training.

    Args:
        image (np.ndarray): Input image.
        width (int): Target width in pixels.
        height (int): Target height in pixels.

    Returns:
        np.ndarray: Resized image with shape (height, width, C).
    """
    return cv2.resize(image, (width, height))


def normalize_for_mobilenet(image: np.ndarray) -> np.ndarray:
    """
    Normalize an RGB image using MobileNetV2 ImageNet statistics.

    Applies the same normalization as in training_pytorch_float.py:
    ``transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])``.

    Args:
        image (np.ndarray): Input RGB image, uint8 [0, 255] or float32 [0.0, 1.0],
            shape (H, W, 3).

    Returns:
        np.ndarray: Normalized image as float32, shape (H, W, 3).
    """
    img_float = image.astype(np.float32) / 255.0 if image.max() > 1.0 else image.astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (img_float - mean) / std


def add_batch_dimension(image: np.ndarray) -> np.ndarray:
    """
    Add a batch dimension to an image: (H, W, C) → (1, H, W, C).

    Required by TFLite interpreters that expect a batch axis as the first dimension.

    Args:
        image (np.ndarray): Image array of shape (H, W, C) or (H, W).

    Returns:
        np.ndarray: Array with an extra leading dimension.
    """
    return np.expand_dims(image, axis=0)


def apply_aggressive_background_mask(image: np.ndarray) -> np.ndarray:
    """
    Mask all background pixels (gray areas) to black, leaving only the black part visible.
    
    Uses adaptive thresholding to separate the black part from the gray background,
    then applies morphological operations to clean up noise and ensure a clean mask.
    
    This is designed to eliminate illumination variations in the background that could
    confuse anomaly detection algorithms like PaDiM.
    
    Args:
        image (np.ndarray): Input RGB image, uint8 format, shape (H, W, 3).
    
    Returns:
        np.ndarray: Image with background masked to black, same shape and dtype as input.
    """
    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    
    # Otsu thresholding: separates black part (low values) from gray background (higher values)
    # The black part will be 0 in the binary mask, background will be 255
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert: we want the black part as 255 (foreground), background as 0
    mask = cv2.bitwise_not(binary_mask)
    
    # Morphological operations to clean up noise and smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # Remove small noise
    
    # Apply mask: keep only the black part, set background to black (0, 0, 0)
    if image.ndim == 3:
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        result = np.where(mask_3ch == 255, image, 0)
    else:
        result = np.where(mask == 255, image, 0)
    
    return result.astype(image.dtype)


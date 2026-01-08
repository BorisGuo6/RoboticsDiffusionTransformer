import numpy as np
from sensor_msgs.msg import Image
import cv2

def bgr_to_rgb(image):
    return image[..., ::-1]  # Reverse the last dimension

def imgmsg_to_cv2(img_msg):
    """
    Convert a ROS Image message to an OpenCV image without using cv_bridge
    
    Args:
        img_msg: ROS Image message
    Returns:
        cv_image: OpenCV image
    """
    # Get the dimensions and encoding
    height = img_msg.height
    width = img_msg.width
    channels = 3
    # Get raw image bytes
    dtype = np.uint8
    
    # Convert bytes to numpy array
    img_buf = np.frombuffer(img_msg.data, dtype=dtype)
    
    # Reshape array to image dimensions
    img_bgr = img_buf.reshape((height, width, channels))

    img_rgb = bgr_to_rgb(img_bgr)

    img_rgb_resize = img_pad_resize(img_rgb)

    # Add visualization
    # visualize_processing(img_bgr, img_rgb_resize)
        
    return img_rgb_resize



def img_pad_resize(rgb_image, target_size=(384, 384)):
    # Image is already in RGB format
    h, w = rgb_image.shape[:2]
    longest_side = max(w, h)
    delta_w = longest_side - w
    delta_h = longest_side - h

    # Create padding values
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # Add padding (black padding)
    padded_image = cv2.copyMakeBorder(
        rgb_image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    # Resize to target size
    resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_LINEAR)

    # cv2.imshow('Resized Image', resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return resized_image

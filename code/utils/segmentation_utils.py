# import basic modules
from PIL import Image, ImageDraw
import numpy as np

def get_crop_box_pixels(crop_box_raw, orig_size):
    
    """
    Convert cropping box from its raw format (from the NSD metadata array) to a bbox in pixels.
    This provides the cropping bbox that is used to get from full COCO image (often rectangle) to the NSD version (always a square)
    Returns [hmin, hmax, wmin, wmax] where h is the "height" dim and w is the "width" dim.
    """
    
    orig_height, orig_width = orig_size
    
    crop_box_pixels = crop_box_raw * np.array([orig_height, orig_height, orig_width, orig_width])
    crop_box_pixels[1] = orig_height - crop_box_pixels[1]
    crop_box_pixels[3] = orig_width - crop_box_pixels[3]    
    
    return np.floor(crop_box_pixels).astype('int')

def crop_to_square(image_array):
 
    # crop rectangular image to a square
    # smaller side becomes the size of final square
    # taking an even amount off each end of the longer side.  
    # image array should be [height, width, ...]
    
    orig_size = image_array.shape[0:2]
    height, width = orig_size
    
    if height>width:
        pct_crop = ((width/height)-1)/2
        bbox_raw = [np.abs(pct_crop), np.abs(pct_crop),0,0]
    else:
        pct_crop = ((height/width)-1)/2
        bbox_raw = [0,0,np.abs(pct_crop), np.abs(pct_crop)]

    crop_box_pixels = get_crop_box_pixels(bbox_raw, orig_size)
   
    cropped = image_array[crop_box_pixels[0]:crop_box_pixels[1], \
                                crop_box_pixels[2]:crop_box_pixels[3]]

    return cropped, bbox_raw


def apply_mask_from_poly(image, polygon_coords, mask_bg_value=0.0):
    
    """ 
    Set to a fixed value all values outside the specified polygon region.
    Polygon is list of pts that goes like [x1, y1, x2, y2 ...] where [x1, y1] specifies coords of first vertex, etc.
    Image is [batch_size x nchannels x height x width] OR [height x width] OR [height x width x 3]
    mask_bg_value is the numerical value assigned to the pixels outside the mask.
    """
    
    if isinstance(polygon_coords, np.ndarray):
        polygon_coords = list(polygon_coords)

    single_image=False
    if len(image.shape)==2:
        image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
        single_image=True
    elif len(image.shape)==3:
        assert(image.shape[2]==3)
        image = np.moveaxis(np.expand_dims(image, axis=0), [3], [1])
        single_image=True
      
    height = image.shape[2]    
    width = image.shape[3]

    mask = Image.new('L', (width, height), 0)
    ImageDraw.Draw(mask).polygon(polygon_coords, outline=1, fill=1)
    mask = np.array(mask.getdata()).reshape((height, width)).astype('float')
    mask[mask==0] = np.nan

    image_masked = image * np.tile(np.expand_dims(np.expand_dims(mask, axis=0),axis=0), [image.shape[0], image.shape[1], 1, 1])
    image_masked[np.isnan(image_masked)] = mask_bg_value

    if single_image:
        image_masked = np.squeeze(image_masked)
        if image_masked.shape[0]==3:
            image_masked = np.moveaxis(image_masked, [0],[2])
        
    return image_masked

def round_polygon_bbox_to_largest(polygon):
    
    assert(polygon[0]==polygon[6] and polygon[1]==polygon[3] and polygon[2]==polygon[4] and polygon[5]==polygon[7])

    polygon = [np.floor(polygon[0]), np.floor(polygon[1]), np.ceil(polygon[2]), np.floor(polygon[3]), np.ceil(polygon[4]), np.ceil(polygon[5]), np.floor(polygon[6]), np.ceil(polygon[7])]
    
    return np.array(polygon).astype('int')
    

def polygon_from_bbox(bbox):
    """
    Input is bbox [xmin, ymin, width, height]
    Output is list of pts that goes like [x1, y1, x2, y2 ...] where [x1, y1] specifies coords of first vertex, etc.
    """
    polygon= [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], bbox[0], bbox[1]+bbox[3]]
    
    return polygon


def adjust_polygon_for_crop(polygon_coords, crop_box_pixels, adjusted_size = 425):
    """
    Adjust the definition of a polygon (annotation) to account for cropping between COCO original and NSD cropped images.
    polygon_coords is list of pts that goes like [x1, y1, x2, y2 ...] where [x1, y1] specifies coords of first vertex, etc. 
    crop_box_pixels is [hmin, hmax, wmin, wmax] where h is the "height" dim and w is the "width" dim.
    adjusted_size is the size of NSD image side after crop/resize.
    """

    if isinstance(polygon_coords, list):
        polygon_coords = np.array(polygon_coords)

    polygon_coords[0::2] -= crop_box_pixels[2] # make sure to use the "width" for x coordinate here.
    polygon_coords[1::2] -= crop_box_pixels[0] # and "height" for y coordinate.
    orig_size = crop_box_pixels[1] - crop_box_pixels[0] # orig_size is the size AFTER crop, but before resize
    polygon_coords = polygon_coords * (adjusted_size/orig_size) # now resize
   
    return list(polygon_coords)


def adjust_polygon_for_scale(polygon_coords, orig_size, scaled_size):
    """
    Adjust the definition of a polygon (annotation) to account for re-scaling of feature maps to achieve different spatial frequencies.
    polygon_coords is list of pts that goes like [x1, y1, x2, y2 ...] where [x1, y1] specifies coords of first vertex, etc. 
    """

    if isinstance(polygon_coords, list):
        polygon_coords = np.array(polygon_coords)
    polygon_coords = polygon_coords * (scaled_size/orig_size)
   
    return list(polygon_coords)

   
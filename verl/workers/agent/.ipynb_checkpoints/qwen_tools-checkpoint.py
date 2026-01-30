import numpy as np
import requests
import io
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
import cv2
import inspect
import sys
from PIL import ImageDraw


def fetch_tool_desc() -> str:
    # Get all classes in current module
    tool_classes = []
    current_module = sys.modules[__name__]
    
    for name, obj in inspect.getmembers(current_module):
        # Check if it's a class that inherits from BaseTool and has required attributes
        if (inspect.isclass(obj) and 
            issubclass(obj, BaseTool) and 
            obj != BaseTool):
            tool_classes.append(obj)
    
    tool_prompts = []
    for tool_class in tool_classes:
        tool_prompts.append(str(tool_class().function))
    
    return '\n'.join(tool_prompts)

def fetch_tools(placeholder='<|vision_start|><|image_pad|><|vision_end|>') -> List[BaseTool]:
    # Get all classes in current module
    tools = {}
    current_module = sys.modules[__name__]
    
    for name, obj in inspect.getmembers(current_module):
        # Check if it's a class that inherits from BaseTool and has required attributes
        if (inspect.isclass(obj) and 
            issubclass(obj, BaseTool) and 
            obj != BaseTool):
            tool_instance = obj(placeholder=placeholder)
            tools[tool_instance.name] = tool_instance
    
    return tools
    

class DepthEstimator(BaseTool):
    name = "depth_estimation"
    description = "Depth estimation using DepthAnything model. It returns the depth map of the input image. " \
                  "A colormap is used to represent the depth . It uses Spectral_r colormap. The closer the object , the warmer the color . " \
                  "This tool may help you to better reason about the spatial relationship , like which object is closer to the camera. "
            
    parameters = [
        {
            'name': 'image_id',
            'type': 'int',
            'description': "The ID of the image in the conversation including images from tool start from 0.",
            'required': True,
        }
    ]
    
    def __init__(self, url='http://localhost:9991/predict/color_depth', placeholder='<|vision_start|><|image_pad|><|vision_end|>'):
        self.url = url
        self.placeholder = placeholder
    
    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image = env_state['image'][args['image_id']]
                        
            # Save PIL Image object to in-memory byte stream
            img_byte_arr = io.BytesIO()
            
            # Handle RGBA mode images, convert to RGB to avoid save errors
            if image.mode == 'RGBA':
                image = image.convert('RGB')
                
            image.save(img_byte_arr, format='JPEG')  # Change format if required by model
            img_byte_arr = img_byte_arr.getvalue()

            # Send POST request, upload image data as file
            files = {'file': ('image.jpeg', img_byte_arr, 'image/jpeg')}
            response = requests.post(self.url, files=files)
            response.raise_for_status()  # Raise exception if not 200 OK

            # Read image data from response and create PIL Image object
            returned_image = Image.open(io.BytesIO(response.content))

            return {
                "text": f"The colored depth map for image {args['image_id']}. ",
                "image": [returned_image]
            }
        except KeyError as e:
            return {
                "text": f"Failed to detect object for image {args['image_id']} due to error: Key error: {str(e)}. Please check the tool_call format.",
                "image": None
            }
        except Exception as e:
            return {
                "text": f"Failed to generate depth map for image {args['image_id']} due to error: {str(e)}",
                "image": None
            }
            
class EdgeDetector(BaseTool):
    name = "edge_detection"
    description = "Uses Scharr edge detection to emphasize object contours. " \
                  "This tool helps identify boundaries and shapes in images."
            
    parameters = [
        {
            'name': 'image_id',
            'type': 'int',
            'description': "The ID of the image in the conversation including images from tool start from 0.",
            'required': True,
        }
    ]
    
    def __init__(self, placeholder='<|vision_start|><|image_pad|><|vision_end|>'):
        self.placeholder = placeholder
    
    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image = env_state['image'][args['image_id']]
            
            # Convert PIL image to numpy array for OpenCV processing
            img_np = np.array(image)
            
            # Import cv2 locally to avoid dependency issues
            
            # Convert to grayscale if the image has color channels
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
                
            # Apply Scharr edge detection
            grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
            
            # Calculate gradient magnitude
            magnitude = cv2.magnitude(grad_x, grad_y)
            
            # Normalize to 0-255 for better visualization
            normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Convert back to PIL image
            returned_image = Image.fromarray(normalized)

            return {
                "text": f"The edge map for image {args['image_id']}. ",
                "image": [returned_image]
            }
        except KeyError as e:
            return {
                "text": f"Failed to detect object for image {args['image_id']} due to error: Key error: {str(e)}. Please check the tool_call format.",
                "image": None
            }
        except Exception as e:
            return {
                "text": f"Failed to generate edge map for image {args['image_id']} due to error: {str(e)}",
                "image": None
            }
            
# class Segmentation(BaseTool):
#     name = "segmentation"
#     description = "Use a segmentation model to segment the image, and add colorful masks on the segmented objects. " \
#                   "Mode can be 'auto' (default, segment the entire image) or 'point' (segment the region around a specific point). " \
#                   "DO NOT use this tool to search or detect an object. Better after zooming in the image. " \
            
#     parameters = [
#         {
#             'name': 'image_id',
#             'type': 'int',
#             'description': "The ID of the image in the conversation including images from tool start from 0.",
#             'required': True,
#         },
#         {
#             'name': 'mode',
#             'type': 'str',
#             'description': "Segmentation mode: 'auto' (segment the entire image) or 'point' (segment based on a specific point).",
#             'required': False,
#             'default': 'auto',
#         },
#         {
#             'name': 'point',
#             'type': 'list',
#             'description': "When mode is 'point', specify the [x, y] coordinates of the point to segment around. Only needed with point mode.",
#             'required': False,
#         }
#     ]
    
#     def __init__(self, url='http://localhost:9992/predict/segmentation', placeholder='<|vision_start|><|image_pad|><|vision_end|>'):
#         self.url = url
#         self.placeholder = placeholder
    
#     def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
#         try:
#             image = env_state['image'][args['image_id']]
            
#             mode = args.get('mode', 'auto')
#             point = args.get('point', None)
            
#             if mode not in ['auto', 'point']:
#                 return {
#                     "text": f"Invalid segmentation mode: {mode}. Must be 'auto' or 'point'.",
#                     "image": None
#                 }
                
#             if mode == 'point' and point is None:
#                 return {
#                     "text": f"Point mode requires 'point' parameter with [x, y] coordinates.",
#                     "image": None
#                 }
                        
#             img_byte_arr = io.BytesIO()
            
#             if image.mode == 'RGBA':
#                 image = image.convert('RGB')
                
#             image.save(img_byte_arr, format='JPEG')  
#             img_byte_arr = img_byte_arr.getvalue()

#             files = {'file': ('image.jpeg', img_byte_arr, 'image/jpeg')}
            
#             form_data = {'mode': mode}
            
#             if mode == 'point' and point:
#                 form_data['point'] = f"{point[0]},{point[1]}"
                
#             response = requests.post(self.url, files=files, data=form_data)
#             response.raise_for_status()  
#             returned_image = Image.open(io.BytesIO(response.content))

#             if mode == 'auto':
#                 response_text = f"The segmentation mask for image {args['image_id']}."
#             else:
#                 response_text = f"The segmentation mask for image {args['image_id']} around point {point}."

#             return {
#                 "text": response_text,
#                 "image": [returned_image]
#             }
#         except KeyError as e:
#             return {
#                 "text": f"Failed to segment image {args['image_id']} due to error: Key error: {str(e)}. Please check the tool_call format.",
#                 "image": None
#             }
#         except Exception as e:
#             return {
#                 "text": f"Failed to generate segmentation mask for image {args['image_id']} due to error: {str(e)}",
#                 "image": None
#             }
            
class ZoomIn(BaseTool):
    name = "zoom_in"
    description = "Enlarges specific image regions to highlight intricate details, aiding in answering questions that require close-up inspection of small elements. You can zoom in on regions that objects are likely to appear."

    parameters = [
        {
            'name': 'image_id',
            'type': 'int',
            'description': "The ID of the image in the conversation including images from tool start from 0.",
            'required': True,
        },
        {
            'name': 'bbox',
            'type': 'list',
            'description': "The bounding box to zoom in on. Format: [x1, y1, x2, y2].",
            'required': True,
        },
        {
            'name': 'factor',
            'type': 'float',
            'description': "The magnification factor. Between 1.0 and 2.0. Default 1.0.",
            'required': False,
            'default': 1.0,
        }
    ]
    
    def __init__(self, placeholder='<|vision_start|><|image_pad|><|vision_end|>'):
        self.placeholder = placeholder
    
    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image = env_state['image'][args['image_id']]
            bbox = args['bbox']
            magnification = args.get('factor', 1.0)
            
            if magnification > 2.0:
                magnification = 2.0
            
            # Ensure bbox is a list of four integers
            if len(bbox) != 4 or not all(isinstance(i, (int, float)) for i in bbox):
                raise ValueError("Bounding box must be a list of four integers or floats.")
            
            # Get image size
            img_width, img_height = image.size
            
            # Limit bbox within image range
            x1 = max(0, min(img_width, bbox[0]))
            y1 = max(0, min(img_height, bbox[1]))
            x2 = max(0, min(img_width, bbox[2]))
            y2 = max(0, min(img_height, bbox[3]))
            
            # Ensure x1 < x2, y1 < y2
            if x1 >= x2:
                x1, x2 = max(0, x2 - 1), min(img_width, x1 + 1)
            if y1 >= y2:
                y1, y2 = max(0, y2 - 1), min(img_height, y1 + 1)
            
            # Check aspect ratio, ensure not exceeding 4
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = max(width / max(1, height), height / max(1, width))
            
            if aspect_ratio > 4:
                # Adjust short side to limit aspect ratio to 4
                if width > height:
                    center_y = (y1 + y2) / 2
                    half_height = width / 8
                    y1 = max(0, int(center_y - half_height))
                    y2 = min(img_height, int(center_y + half_height))
                else:
                    center_x = (x1 + x2) / 2
                    half_width = height / 8
                    x1 = max(0, int(center_x - half_width))
                    x2 = min(img_width, int(center_x + half_width))
            
            cropped_image = image.crop((x1, y1, x2, y2))
            
            crop_width, crop_height = cropped_image.size
            new_width = int(crop_width * magnification)
            new_height = int(crop_height * magnification)
            
            zoomed_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            min_dim = min(new_width, new_height)
            if min_dim < 28:
                scale_factor = 28 / min_dim
                final_width = int(new_width * scale_factor)
                final_height = int(new_height * scale_factor)
                
                zoomed_image = zoomed_image.resize((final_width, final_height), Image.Resampling.LANCZOS)

            return {
                "text": f"Zoomed image {args['image_id']} on {bbox} with {magnification}x magnification.",
                "image": [zoomed_image]
            }
        except KeyError as e:
            return {
                "text": f"Failed to detect object for image {args['image_id']} due to error: Key error: {str(e)}. Please check the tool_call format.",
                "image": None
            }
        except Exception as e:
            return {
                "text": f"Failed to zoom in on image {args['image_id']} due to error: {str(e)}",
                "image": None
            }
            
# class Transpose(BaseTool):
#     name = "transpose"
#     description = "Transforms images by flipping or rotating them to aid in viewing from different angles. "
            
#     parameters = [
#         {
#             'name': 'image_id',
#             'type': 'int',
#             'description': "The ID of the image in the conversation including images from tool start from 0.",
#             'required': True,
#         },
#         {
#             'name': 'operation',
#             'type': 'str',
#             'description': "The operation to perform. Options: 'ROTATE_90', 'ROTATE_180', 'ROTATE_270', 'FLIP_LEFT_RIGHT', 'FLIP_TOP_BOTTOM'.",
#             'required': True,
#         }
#     ]
    
#     def __init__(self, placeholder='<|vision_start|><|image_pad|><|vision_end|>'):
#         self.placeholder = placeholder
    
#     def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
#         try:
#             image = env_state['image'][args['image_id']]
            
#             type2id = {
#                 'FLIP_LEFT_RIGHT': 0,
#                 'FLIP_TOP_BOTTOM': 1,
#                 'ROTATE_90': 2,
#                 'ROTATE_180': 3,
#                 'ROTATE_270': 4
#             }
#             # Transpose the image
#             transposed_image = image.transpose(method=Image.Transpose(type2id[args['operation']]))
            
#             return {
#                 "text": f"The transposed image {args['image_id']}. ",
#                 "image": [transposed_image]
#             }
#         except KeyError as e:
#             return {
#                 "text": f"Failed to detect object for image {args['image_id']} due to error: Key error: {str(e)}. Please check the tool_call format.",
#                 "image": None
#             }
#         except Exception as e:
#             return {
#                 "text": f"Failed to transpose image {args['image_id']} due to error: {str(e)}",
#                 "image": None
#             }
            
class ObjectDetection(BaseTool):
    name = "object_detection"
    description = "Object detection using Grounding DINO model. It returns the annotated image and the bounding boxes of the detected objects. " \
                  "The detector is not perfect , it may wrongly detect objects or miss some objects. You should use the output as a reference , not as a ground truth. Better after zooming in the image."
            
    parameters = [
        {
            'name': 'image_id',
            'type': 'int',
            'description': "The ID of the image in the conversation including images from tool start from 0.",
            'required': True,
        },
        {
            'name': 'objects',
            'type': 'List[str]',
            'description': "The objects to detect in the image.",
            'required': True,
        }
        # {
        #     'name': 'display',
        #     'type': 'str',
        #     'description': "The display mode. Options: 'bbox', 'mask', 'both'. Default 'bbox'.",
        #     'required': False,
        # }
    ]
    
    def __init__(self, url='http://localhost:9993/detect', placeholder='<|vision_start|><|image_pad|><|vision_end|>'):
        self.url = url
        self.placeholder = placeholder
    
    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image = env_state['image'][args['image_id']]
            
            # Save PIL Image object to in-memory byte stream
            img_byte_arr = io.BytesIO()
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            # Send POST request, upload image data as file
            objects = (". ".join(args['objects'])+".").lower()
            
            data = {
                "text_prompt": objects,
                "box_threshold": 0.2,
                "text_threshold": 0.3,
            }
            files = {'file': ('image.jpeg', img_byte_arr, 'image/jpeg')}
            response = requests.post(self.url, files=files, data=data)
            response.raise_for_status()
            
            if response.headers.get('content-type') == 'application/json':
                json_response = response.json()
                
                if "message" in json_response:
                    return {
                        "text": f"No objects matching '{objects}' detected in image {args['image_id']}." \
                               "The result of the detection may be wrong, don't treat it as ground truth.",
                        "image": None
                    }
                    
                detected_objects = json_response.get('detected_objects', [])
                detection_text = f"Detected {len(detected_objects)} object(s) in image {args['image_id']}:\n"
                
                for obj in detected_objects:
                    detection_text += f"{obj['id']}. {obj['label']}({obj['confidence']:.2f}): {[int(x) for x in obj['bbox']]}\n"
                detection_text += f"Detection result may be wrong, don't treat it as ground truth."
                    
                visualization_path = json_response.get('visualization_path')
                result_image = None
                
                if visualization_path:
                    try:
                        result_image = Image.open(visualization_path)
                        return {
                            "text": f"{detection_text}",
                            "image": [result_image]
                        }
                    except Exception as e:
                        return {
                            "text": f"{detection_text}\nFailed to load visualization: {str(e)}",
                            "image": None
                        }
                else:
                    return {
                        "text": detection_text,
                        "image": None
                    }
                    
            elif response.headers.get('content-type', '').startswith('image/'):
                result_image = Image.open(io.BytesIO(response.content))
                return {
                    "text": f"Objects detection for image {args['image_id']} ",
                    "image": [result_image]
                }
                
            else:
                return {
                    "text": f"Unexpected response type for image {args['image_id']}. Cannot process result.",
                    "image": None
                }
        except KeyError as e:
            return {
                "text": f"Failed to detect object for image {args['image_id']} due to error: Key error: {str(e)}. Please check the tool_call format.",
                "image": None
            }
        except Exception as e:
            return {
                "text": f"Failed to detect object for image {args['image_id']} due to error: {str(e)}",
                "image": None
            }
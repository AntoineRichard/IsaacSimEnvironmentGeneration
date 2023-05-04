import numpy as np 
import cv2 

def nearest_interpolation(data:np.ndarray, scale:int=1):
    return cv2.resize(data, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def bilinear_interpolation(data:np.ndarray, scale:int=1):
    return cv2.resize(data, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

def bicubic_interpolation(data:np.ndarray, scale:int=1):
    return cv2.resize(data, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
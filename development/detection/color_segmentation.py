import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_red_channel(hsv_image):
    # The red color in HSV in two ranges
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # masks for the red color
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Morphological operation
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    return red_mask

def detect_blue_channel(hsv_image):
    # The blue color in HSV ranges
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # create mask
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Morphological operation
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)  # delete noise
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel) 
    
    return blue_mask

def detect_white_channel(bgr_image, hsv_image):
    # Разделяем каналы BGR для проверки яркости
    b, g, r = cv2.split(bgr_image)
    
    # Проверка яркости (Value > 200) в HSV
    v_channel = hsv_image[:, :, 2]
    brightness_mask = (v_channel > 200).astype(np.uint8) * 255
    
    # Проверка насыщенности (Saturation < 50) в HSV
    s_channel = hsv_image[:, :, 1]
    saturation_mask = (s_channel < 50).astype(np.uint8) * 255
    
    # Объединяем условия: высокая яркость И низкая насыщенность
    white_mask = cv2.bitwise_and(brightness_mask, saturation_mask)
    
    return white_mask

# список всех подходящих контуров 
def process_all_contours(masks_dict, contour_params, min_area=600, max_area=6000):
    
    max_aspect_ratio = contour_params.get('max_aspect_ratio', 4.0)
    epsilon_factor = contour_params.get('epsilon_factor', 0.02)
    
    all_contours = []
    
    for color_name, mask in masks_dict.items():
        if mask is None:
            continue
            
        # Поиск всех контуров на маске
        contours, hierarchy = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            # Вычисляем характеристики контура
            area = cv2.contourArea(contour)
            
            # Пропускаем контуры с нулевой площадью
            if area == 0:
                continue
                
            # Ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
            
            # Фильтрация по параметрам
            if (area < min_area or 
                area > max_area or 
                aspect_ratio > max_aspect_ratio):
                continue
            
            # Аппроксимация контура
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Просто добавляем контур в список
            all_contours.append(approx_contour)
    
    return all_contours

# =========== MAIN process function ============
def process_image(image):
    # BGR -> HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Detection of the colors 
    red_mask = detect_red_channel(hsv_image)
    blue_mask = detect_blue_channel(hsv_image)
    white_mask = detect_white_channel(image, hsv_image)
    
    contour_params = {
        'min_area': 600,
        'max_area': 60000,
        'max_aspect_ratio': 4.0,
        'epsilon_factor': 0.02
    }
    
    masks_dict = {
        'red': red_mask,
        'blue': blue_mask,
        'white': white_mask
    }

    contours_result = process_all_contours(masks_dict, contour_params, 600, 60000)
    
    result = {
        'original': image,
        'red_mask': red_mask,
        'blue_mask': blue_mask,
        'white_mask': white_mask,
        'hsv_image': hsv_image,
        'contours_result': contours_result
    }
    
    return result





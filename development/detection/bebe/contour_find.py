import cv2
import numpy as np

def process_contours(mask, color_name, min_area=600, max_area=57000, max_aspect_ratio=4.0, epsilon_factor=0.02):
    """
    Параметры:
    - mask: бинарная маска
    - color_name: название цвета для отладки
    - min_area: минимальная площадь контура
    - max_area: максимальная площадь контура  
    - max_aspect_ratio: максимальное соотношение сторон
    - epsilon_factor: коэффициент для аппроксимации контура

    Возвращает:
    - filtered_contours: отфильтрованные контуры
    """

    
    # 3.1. Find all contours 
    contours, hierarchy = cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    
    filtered_contours = []
    # contour_info = []
    
    for i, contour in enumerate(contours):
        # Вычисляем характеристики контура
        area = cv2.contourArea(contour)
        
        # Пропускаем контуры с нулевой площадью
        if area == 0:
            continue
            
        # Ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
        
        # 3.2. filtr 
        if area < min_area or area > max_area or aspect_ratio > max_aspect_ratio :
            continue
        
        # 3.3. approxPolyDP
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Сохраняем отфильтрованный контур
        filtered_contours.append(approx_contour)
        
        # Сохраняем информацию о контуре
        # contour_info.append({
        #     'original_contour': contour,
        #     'approx_contour': approx_contour,
        #     'area': area,
        #     'bounding_rect': (x, y, w, h),
        #     'aspect_ratio': aspect_ratio,
        #     'num_vertices': len(approx_contour),
        #     'center': (x + w//2, y + h//2)
        # })
        
    
    
    return filtered_contours

# def draw_contours_results(image, contours_info, color_name, color_bgr):
#     """
#     Визуализация результатов обработки контуров
    
#     Параметры:
#     - image: исходное изображение
#     - contours_info: информация о контурах
#     - color_name: название цвета
#     - color_bgr: цвет в формате BGR
    
#     Возвращает:
#     - result_image: изображение с визуализацией
#     """
#     result_image = image.copy()
    
#     for i, info in enumerate(contours_info):
#         # Рисуем аппроксимированный контур
#         cv2.drawContours(result_image, [info['approx_contour']], -1, color_bgr, 3)
        
#         # Рисуем bounding box
#         x, y, w, h = info['bounding_rect']
#         cv2.rectangle(result_image, (x, y), (x + w, y + h), color_bgr, 2)
        
#         # Рисуем центр
#         center_x, center_y = info['center']
#         cv2.circle(result_image, (center_x, center_y), 5, color_bgr, -1)
        
#         # Добавляем информацию
#         label = f"{color_name} {i}: A={info['area']:.0f}, V={info['num_vertices']}"
#         cv2.putText(result_image, label, (x, y - 10), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
    
#     return result_image


def process_all_contours(masks_dict, contour_params, min_area, max_area):
    """
    Обработка контуров для всех масок
    
    Параметры:
    - masks_dict: словарь с масками {'red': red_mask, 'blue': blue_mask, ...}
    - contour_params: словарь с параметрами обработки контуров
    
    Возвращает:
    - contours_result: словарь с результатами обработки контуров
    """
    contours_result = {}
    
    # Цвета для визуализации (BGR format)
    colors = {
        'red': (0, 0, 255),
        'blue': (255, 0, 0), 
        'white': (255, 255, 255)
    }
    
    for color_name, mask in masks_dict.items():
        if mask is not None:
            contours = process_contours(
                mask, 
                color_name,
                min_area=contour_params.get('min_area', min_area),
                max_area=contour_params.get('max_area', max_area),
                max_aspect_ratio=contour_params.get('max_aspect_ratio', 4.0),
                epsilon_factor=contour_params.get('epsilon_factor', 0.02)
            )
            contours_result[color_name] = {
                'contours': contours,
                'color_bgr': colors.get(color_name, (0, 255, 0))
            }
    
    return contours_result







 # НУЖНО ТОЛЬКО ДЛЯ ОТЛАДКИ 

# def create_all_contours_image(image, contours_result):
#     """
#     Создание изображения со всеми контурами
    
#     Параметры:
#     - image: исходное изображение
#     - contours_result: результаты обработки контуров
    
#     Возвращает:
#     - all_contours_image: изображение со всеми контурами
#     """
#     all_contours_image = image.copy()
    
#     for color_name, result in contours_result.items():
#         for info in result['info']:
#             cv2.drawContours(all_contours_image, [info['approx_contour']], -1, result['color_bgr'], 2)
    
#     return all_contours_image


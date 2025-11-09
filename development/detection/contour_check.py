import cv2
import numpy as np
from typing import Dict, List, Tuple, Any

def analyze_contours(contours_result: Dict[str, Any], 
                    min_size: Tuple[int, int] = (20, 20),
                    corner_margin: int = 50,
                    min_overlap_distance: int = 30) -> Dict[str, Any]:
    """
    Анализирует контуры и возвращает валидные ROI
    """
    image = contours_result['original']
    contours = contours_result.get('contours', [])
    height, width = image.shape[:2]
    
    valid_candidates = []
    
    for i, contour in enumerate(contours):
        # Анализ формы и геометрии
        shape_info = analyze_shape(contour)
        context_info = analyze_context(contour, width, height, corner_margin)
        
        # Проверка всех критериев
        if all([shape_info['shape_valid'],  context_info['context_valid']]):
            candidate = {
                'contour_id': i,
                'contour': contour,
                **shape_info,
                **context_info
            }
            valid_candidates.append(candidate)
    
    # Удаляем перекрывающиеся ROI
    valid_candidates = remove_wrong_roi(valid_candidates, min_overlap_distance)
    
    # Извлекаем ROI
    valid_rois = extract_rois(image, valid_candidates)
    
    return {
        'original_image': image,
        'valid_candidates': valid_candidates,
        'valid_rois': valid_rois,
        'total_contours': len(contours),
        'valid_contours': len(valid_candidates)
    }

def analyze_shape(contour: np.ndarray) -> Dict[str, Any]:

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Approximation  
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    
    # Roundary
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    

    shape_valid = (
        vertices in [3, 4] or vertices >= 6 and  # vertices
        0.2 <= circularity <= 1.0 and            #   roundary
        solidity >= 0.7                          # Solidity
    )
    
    return {
        'area': area,
        'vertices': vertices,
        'circularity': circularity,
        'solidity': solidity,
        'shape_valid': shape_valid
    }


def analyze_context(contour: np.ndarray, img_w: int, img_h: int, margin: int) -> Dict[str, Any]:
    
    x, y, w, h = cv2.boundingRect(contour)
    center_x, center_y = x + w//2, y + h//2
    
    
    in_corner = any([
        center_x < margin and center_y < margin,                       # left up
        center_x > img_w - margin and center_y < margin,              # right up 
        center_x < margin and center_y > img_h - margin,              # left down
        center_x > img_w - margin and center_y > img_h - margin       # right down
    ])
    
    return {
        'center': (center_x, center_y),
        'context_valid': not in_corner
    }

def remove_wrong_roi(candidates: List[Dict], min_distance: int) -> List[Dict]:
    
    if len(candidates) <= 1:
        return candidates
    
    # Sort
    candidates.sort(key=lambda x: x['area'], reverse=True)
    filtered = []
    
    for candidate in candidates:
        x1, y1, w1, h1 = candidate['bounding_rect']
        center1 = np.array([x1 + w1/2, y1 + h1/2])
        
        too_close = any(
            np.linalg.norm(center1 - np.array([x2 + w2/2, y2 + h2/2])) < min_distance
            for x2, y2, w2, h2 in [c['bounding_rect'] for c in filtered]
        )
        
        if not too_close:
            filtered.append(candidate)
    
    return filtered

def extract_rois(image: np.ndarray, candidates: List[Dict]) -> List[Dict]:

    rois = []
    
    for candidate in candidates:
        x, y, w, h = candidate['bounding_rect']
        
        # Image shape
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            continue
            
        roi_image = image[y:y+h, x:x+w].copy()
        
        rois.append({
            'contour_id': candidate['contour_id'],
            'roi_image': roi_image,
            'bounding_rect': (x, y, w, h)
        })
    
    return rois

def visualize_analysis(image: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    """Визуализация результатов"""
    vis_image = image.copy()
    
    for candidate in analysis['valid_candidates']:
        contour = candidate['contour']
        x, y, w, h = candidate['bounding_rect']
        
        # Рисуем контур и bounding box
        cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(vis_image, f"ID:{candidate['contour_id']}", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_image



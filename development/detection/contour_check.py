import cv2
import numpy as np

def analyze_contours(
    contours,
    image_shape,
    min_size=(10, 10),
    aspect_ratio_range=(0.5, 2.0),
    roundness_threshold=0.3,
    solidity_threshold=0.8,
    corner_margin=10,
    overlap_threshold=0.5
):

    def ensure_contour_format(contour):
        """Приводит контур к правильному формату для OpenCV."""
        if contour is None:
            return None
            
        if not isinstance(contour, np.ndarray):
            contour = np.array(contour, dtype=np.int32)

        if len(contour.shape) == 2 and contour.shape[1] == 2:
            contour = contour.reshape(-1, 1, 2)
        elif len(contour.shape) == 1:
            if len(contour) % 2 == 0:
                contour = contour.reshape(-1, 1, 2)
            else:
                return None
        
        return contour.astype(np.int32)

    def check_shape(contour):
        if contour is None or len(contour) < 3:
            return False
        
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        n = len(approx)
        
    
        if n in [3, 4]:
            return True
        
        if n >= 6 and cv2.isContourConvex(approx):
            sides = [np.linalg.norm(approx[(i+1)%n][0] - approx[i][0]) for i in range(n)]
            avg_side = np.mean(sides)
            return all(abs(side - avg_side) < 0.3 * avg_side for side in sides)
        
        return False

    def calculate_shape_features(contour): 
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        approx = cv2.approxPolyDP(contour,  0.02 * perimeter, True)
        
        roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        type_of_shape = check_shape(contour) 
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'vertices_count': len(approx),
            'roundness': roundness,
            'convexity': cv2.isContourConvex(contour),
            'solidity': solidity,
            'approx_contour': approx,
            'type_of_shape': type_of_shape
        }
    
    def calculate_overlap(rect1, rect2):

        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        dx = min(x1 + w1, x2 + w2) - max(x1, x2)
        dy = min(y1 + h1, y2 + h2) - max(y1, y2)
        
        if dx <= 0 or dy <= 0:
            return 0.0
        
        intersection_area = dx * dy
        area1 = w1 * h1
        area2 = w2 * h2
        
        
        return intersection_area / min(area1, area2)

    if contours is None or len(contours) == 0:
        return []
    
    valid_contours = []  #
    contours_data = []
    img_height, img_width = image_shape
    min_width, min_height = min_size
    min_ar, max_ar = aspect_ratio_range
    
    for i, contour in enumerate(contours):

        formatted_contour = ensure_contour_format(contour)
        if formatted_contour is None:
            continue
            
    
        shape_features = calculate_shape_features(formatted_contour)
        if shape_features is None:
            continue
            
        try:
            x, y, w, h = cv2.boundingRect(formatted_contour)
            aspect_ratio = w / h if h > 0 else 0
            
            geometry_features = {
                'bounding_rect': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'meets_size_requirements': w >= min_width and h >= min_height,
                'width': w,
                'height': h,
                'center': (x + w // 2, y + h // 2)
            }
            
            contours_data.append({
                'contour': formatted_contour, 
                'shape': shape_features,
                'geometry': geometry_features,
                'index': i
            })
        except Exception as e:
            print(f"error with countor {i}: {e}")
            continue
    
    if not contours_data:
        return []
    
    contours_data.sort(key=lambda c: cv2.contourArea(c['contour']))
    for i, contour_data in enumerate(contours_data):
        rect = contour_data['geometry']['bounding_rect']
        x, y, w, h = rect
        
        in_corner = (
            x < corner_margin or 
            y < corner_margin or 
            x + w > img_width - corner_margin or 
            y + h > img_height - corner_margin
        )
        
        max_overlap = 0.0
        has_significant_overlap = False
        
        for j, other_data in enumerate(contours_data[i+1:]):
                
            X, Y, W, H = other_data['geometry']['bounding_rect']
            if (x >= X and y >= Y and x + w <= X + W and y + h <= Y + H):
                has_significant_overlap = True
                max_overlap = 1.0
                break

            overlap = calculate_overlap(rect, other_data['geometry']['bounding_rect'])
            max_overlap = max(max_overlap, overlap)
            
            if overlap > overlap_threshold:
                has_significant_overlap = True
                break
        
        contour_data['context'] = {
            'in_corner': in_corner,
            'has_significant_overlap': has_significant_overlap,
            'overlap_score': max_overlap
        }
    
    for contour_data in contours_data:
        shape = contour_data['shape']
        geometry = contour_data['geometry']
        context = contour_data['context']
        
        if (shape['convexity'] and 
            shape['roundness'] >= roundness_threshold and
            # shape['solidity'] >= solidity_threshold and
            geometry['meets_size_requirements'] and
            # not context['in_corner'] and
            not context['has_significant_overlap'] and
            min_ar <= geometry['aspect_ratio'] <= max_ar and 1):
            # shape['type_of_shape']):
            valid_contours.append(contour_data['contour'])
    
    
    if valid_contours:
        valid_contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    
    return valid_contours
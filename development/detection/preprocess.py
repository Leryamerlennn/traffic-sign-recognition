import cv2 
import numpy as np

def resize_frm(frame, target_size = (640, 640)):
    h, w = frame.shape[:2]
    target_h, target_w = target_size

    scale = min((target_w / w), (target_h / h))

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
    
    result = np.zeros((target_h, target_w, 3), dtype = np.uint8)

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return result


def apply_clahe(frame, clip_limit=2.0, grid_size=(8, 8)):
    """Применение CLAHE для улучшения контраста"""
    # BGR -> LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Разделяем каналы
    l, a, b = cv2.split(lab)
    
    # Применяем CLAHE к L-каналу (яркость)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_clahe = clahe.apply(l)
    
    # Объединяем каналы обратно
    lab_clahe = cv2.merge([l_clahe, a, b])
    
    # Конвертируем обратно в BGR
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return result


def correct_illumination(frame, alpha=1.0, beta=0):
    """Коррекция освещения через гамма-коррекцию и выравнивание яркости"""
    # BGR -> GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    otsu_threshold, image_result = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    gamma = max(0.5, 1.0 - (otsu_threshold - 100) / 255.0)
    inv_gamma = 1.0 / gamma

    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Применяем коррекцию
    return cv2.LUT(frame, table)
    
    return 



def preprocessing(frame):
    #resize to the fix size 
    resized = resize_frm(frame, target_size=(640, 640))
    
    # 1.2. Normalize histogramm 
    contrast_enhanced = apply_clahe(resized)
    
    # 1.3. GaussianBlur
    blurred = cv2.GaussianBlur(contrast_enhanced, (3,3), 0)

    # 1.4.Correct Illumination
    final_image = correct_illumination(blurred)
    
    return final_image


# # Пример использования
# if __name__ == "__main__":
#     # Загрузка изображения
#     frame = cv2.imread("/Users/anastasia/Learning/traffic-sign-recognition/data/example/tipy-dorozhnyh-znakov-00-min.jpg")
    
#     if frame is not None:
#         # Применение алгоритма предобработки
#         processed_image = preprocessing(frame)
        
#         # Сохранение результата
#         #cv2.imwrite("processed_image.jpg", processed_image)
        
#         # Показать оригинал и результат
#         cv2.imshow("Original", frame)
#         cv2.imshow("Processed", processed_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print("Ошибка: Не удалось загрузить изображение")
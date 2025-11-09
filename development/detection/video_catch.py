import cv2
import os
import numpy as np
from datetime import datetime
from preprocess import preprocessing
from color_segmentation import process_image
from contour_check import analyze_contours, visualize_analysis

def camera_work():
#=============================

    # Создаем папку для сохранения результатов
    output_dir = "camera_analysis_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Создаем подпапку с временной меткой для текущей сессии
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(output_dir, f"session_{session_time}")
    os.makedirs(session_dir)
    print(f"Session directory: {session_dir}")


#=============================



    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("Error: The camera is not connected")
        return 
    

    print("The camera is connected. Press the ESC key to exit")
    
    tm = cv2.TickMeter()
    tm.start()
    
    fps = 0
    count = 0

    while True:
        
        ret, frame = video.read()

        if not ret: 
            print("Error: The frame was not read.")
            break 


        

        #вызов функций 
        blck = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = preprocessing(frame)
        result1 = process_image(result)
        
        # Анализируем контуры
        analysis_result = analyze_contours(process_image(result))

        cv2.imshow('Webcam', frame)
        #Если захочу проверить 

        contours_result = result1['red_mask']
        cv2.imshow('Webcam', contours_result)

        # ВИЗУАЛИЗАЦИЯ В РЕАЛЬНОМ ВРЕМЕНИ
        visualize_contour_analysis(frame, analysis_result)




        # Exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

        count += 1
        if count % 50 == 0:
            tm.stop()
            fps = 50 / tm.getTimeSec()  
            tm.reset()
            tm.start()

        # FPS 
        print( f"FPS: {fps:.2f}")
    
    # DELETE 
    video.release()
    cv2.destroyAllWindows()



def visualize_contour_analysis(original_image, analysis_result):
    """
    Визуализирует результаты анализа контуров в реальном времени
    
    Args:
        original_image: исходное изображение с камеры
        analysis_result: результат функции analyze_contours()
    """
    # Создаем копию изображения для рисования
    visualization = original_image.copy()
    
    # Получаем данные из анализа
    filtered_contours = analysis_result.get('filtered_contours', [])
    contour_properties = analysis_result.get('contour_properties', [])
    shape_types = analysis_result.get('shape_types', [])
    
    # Цвета для разных типов фигур
    color_map = {
        'rectangle': (0, 255, 0),      # Зеленый
        'triangle': (255, 0, 0),       # Синий
        'circle': (0, 0, 255),         # Красный
        'unknown': (255, 255, 0),      # Голубой
        'polygon': (255, 0, 255)       # Пурпурный
    }
    
    # Рисуем все найденные контуры
    for i, contour in enumerate(filtered_contours):
        if len(contour_properties) > i:
            props = contour_properties[i]
            shape_type = shape_types[i] if i < len(shape_types) else 'unknown'
            
            # Выбираем цвет в зависимости от типа фигуры
            color = color_map.get(shape_type, (255, 255, 0))
            
            # Рисуем контур
            cv2.drawContours(visualization, [contour], -1, color, 2)
            
            # Рисуем ограничивающий прямоугольник
            x, y, w, h = props['bounding_rect']
            cv2.rectangle(visualization, (x, y), (x + w, y + h), color, 1)
            
            # Отображаем центр масс
            cx, cy = props['centroid']
            cv2.circle(visualization, (int(cx), int(cy)), 3, color, -1)
            
            # Добавляем текст с информацией
            info_text = f"{shape_type} A:{props['area']:.0f}"
            cv2.putText(visualization, info_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Добавляем общую статистику
    stats_text = f"Contours found: {len(filtered_contours)}"
    cv2.putText(visualization, stats_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Показываем изображение с визуализацией
    cv2.imshow('Contour Analysis', visualization)
    
    return visualization


if __name__ == "__main__":
    camera_work()
    
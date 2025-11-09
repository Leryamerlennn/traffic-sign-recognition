import cv2
import numpy as np
import os

class SimpleTrafficSignDetector:

    
    def __init__(self):
        # Define color ranges for traffic sign detection
        self.red_lower = np.array([100, 0, 0])    # Minimum red values
        self.red_upper = np.array([255, 50, 50])  # Maximum red values
        
        self.blue_lower = np.array([0, 0, 100])   # Minimum blue values
        self.blue_upper = np.array([50, 50, 255]) # Maximum blue values
        
        # Minimum contour area to filter out small noise
        self.min_contour_area = 500

    def create_color_mask(self, frame):
        """Create a mask based on RGB color ranges"""
        red_mask = cv2.inRange(frame, self.red_lower, self.red_upper)
        blue_mask = cv2.inRange(frame, self.blue_lower, self.blue_upper)
        combined_mask = cv2.bitwise_or(red_mask, blue_mask)
        return combined_mask

    def find_contours(self, mask):
        """Find contours in the binary mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                significant_contours.append(contour)
                
        return significant_contours

    def detect_shapes(self, contours):
        """Detect geometric shapes from contours"""
        detected_shapes = []
        
        for contour in contours:
            # Approximate contour to reduce number of points
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            vertices_count = len(approximated_contour)
            
            # Classify shape based on number of vertices
            if vertices_count == 3:
                shape_type = "triangle"
            elif vertices_count == 4:
                shape_type = "rectangle"
            elif vertices_count > 6:
                shape_type = "circle"
            else:
                shape_type = "unknown"
            
            if shape_type != "unknown":
                detected_shapes.append({
                    'shape': shape_type,
                    'contour': approximated_contour
                })
                
        return detected_shapes

    def detect_image(self, image_path):
        """Main detection function for a single image"""
        
        # Construct full path to image
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        full_image_path = os.path.join(project_root, "data", "example", image_path)

        image = cv2.imread(full_image_path)
    

        color_mask = self.create_color_mask(image)
        contours = self.find_contours(color_mask)
        shapes = self.detect_shapes(contours)
        
        # Draw detection results on image
        result_image = image.copy()
        for shape_info in shapes:
            contour = shape_info['contour']
            shape_name = shape_info['shape']
            
            # Draw contour and bounding box
            cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x + width, y + height), (255, 0, 0), 2)
            
            # Add shape label
            cv2.putText(result_image, shape_name, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image, shapes

    def show_result(self, result_frame, shapes):
        """Show the results, just to show, in the final program we will give only the object"""
        if result_frame is None:
            print("No image to display")
            return
            
        # Print detection summary
        if shapes:
            print(f"Found {len(shapes)} objects:")
            for i, shape_info in enumerate(shapes):
                print(f"  Object {i+1}: {shape_info['shape']}")
        else:
            print("No objects detected")
        
        # Show image
        cv2.imshow('Traffic Sign Detection Results', result_frame)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_traffic_signs(image_path):
    """Convenience function for quick traffic sign detection"""
    detector = SimpleTrafficSignDetector()
    result_image, detected_shapes = detector.detect_image(image_path)
    
    if result_image is not None:
        detector.show_result(result_image, detected_shapes)
    
    return result_image, detected_shapes


# TRY
if __name__ == "__main__":

    test_image_path = "347424594092648.jpeg"
    
    result_image, detected_objects = detect_traffic_signs(test_image_path)

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from keras.models import load_model
from detection.detector import SimpleTrafficSignDetector

# Define traffic sign classes based
TRAFFIC_SIGN_CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vechiles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vechiles',
    16: 'Vechiles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vechiles over 3.5 metric tons',
}


class TrafficSignRecognitionSystem:
    """Combined detection and classification system for traffic signs"""
    
    def __init__(self, model_path=None):
        """Initialize the system with detector and classifier"""
        # Initialize detector
        self.detector = SimpleTrafficSignDetector()
        
        # Load classification model
        if model_path is None:
            # Default path to the model
            current_dir = Path(__file__).parent
            model_path = current_dir / 'classification' / 'model' / 'traffic_signs_cnn_final.keras'
        
        self.model_path = model_path
        try:
            self.model = load_model(str(model_path))
            print(f"✓ Model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
    
    def preprocess_sign(self, image, target_size=(32, 32)):
        """Preprocess a detected sign for classification"""
        # Resize to target size
        resized = cv2.resize(image, target_size)
        
        # Normalize to 0-1 range
        normalized = resized.astype('float32') / 255.0
        
        # Add batch dimension
        batch = np.expand_dims(normalized, axis=0)
        
        return batch
    
    def classify_sign(self, sign_image):
        """Classify a detected traffic sign"""
        if self.model is None:
            return "Unknown", 0.0
        
        try:
            # Preprocess the image
            processed = self.preprocess_sign(sign_image)
            
            # Make prediction
            predictions = self.model.predict(processed, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            
            # Get class name
            class_name = TRAFFIC_SIGN_CLASSES.get(class_idx, f'Class {class_idx}')
            
            return class_name, confidence
        except Exception as e:
            print(f"Error during classification: {e}")
            return "Error", 0.0
    
    def process_frame(self, frame):
        """Process a single frame: detect signs and classify them"""
        if frame is None:
            return None, []
        
        # Detect objects
        color_mask = self.detector.create_color_mask(frame)
        contours = self.detector.find_contours(color_mask)
        shapes = self.detector.detect_shapes(contours)
        
        # Process detections
        detections = []
        result_frame = frame.copy()
        
        for shape_info in shapes:
            contour = shape_info['contour']
            
            # Get bounding box
            x, y, width, height = cv2.boundingRect(contour)
            
            # Extract the region of interest (sign)
            sign_roi = frame[y:y+height, x:x+width]
            
            if sign_roi.size == 0:
                continue
            
            # Classify the sign
            class_name, confidence = self.classify_sign(sign_roi)
            
            detection_info = {
                'bbox': (x, y, width, height),
                'class': class_name,
                'confidence': confidence,
                'shape': shape_info['shape'],
                'contour': contour
            }
            detections.append(detection_info)
            
            # Draw on frame
            result_frame = self._draw_detection(
                result_frame, x, y, width, height, 
                class_name, confidence
            )
        
        return result_frame, detections
    
    def _draw_detection(self, frame, x, y, width, height, class_name, confidence):
        """Draw bounding box and classification label on frame"""
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Draw background rectangle for text
        text_x = x
        text_y = y - 10
        cv2.rectangle(
            frame,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y),
            (0, 255, 0),
            -1
        )
        
        # Put text
        cv2.putText(
            frame, label,
            (text_x, text_y),
            font, font_scale,
            (0, 0, 0), thickness
        )
        
        return frame
    
    def process_video(self, video_source=None, output_path=None, confidence_threshold=0.5):
        """Process video or camera stream"""
        # Set up video source
        if video_source is None or video_source == 0:
            cap = cv2.VideoCapture(0)
            source_name = "Camera"
        elif isinstance(video_source, int):
            cap = cv2.VideoCapture(video_source)
            source_name = f"Camera {video_source}"
        else:
            cap = cv2.VideoCapture(video_source)
            source_name = f"Video: {video_source}"
        
        if not cap.isOpened():
            print(f"✗ Cannot open {source_name}")
            return
        
        print(f"✓ Processing {source_name}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set up video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"✓ Output will be saved to: {output_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print(f"✓ Processing completed. Total frames: {frame_count}")
                    break
                
                frame_count += 1
                
                # Process frame
                processed_frame, detections = self.process_frame(frame)
                
                # Filter by confidence threshold
                filtered_detections = [d for d in detections if d['confidence'] >= confidence_threshold]
                
                # Display frame info
                info_text = f"Frame: {frame_count} | Detections: {len(filtered_detections)}"
                cv2.putText(
                    processed_frame, info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255), 2
                )
                
                # Display frame
                cv2.imshow(source_name, processed_frame)
                
                # Save frame if writer is initialized
                if writer:
                    writer.write(processed_frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("✓ Processing stopped by user")
                    break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
    
    def process_image(self, image_path, output_path=None):
        """Process a single image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ Cannot read image: {image_path}")
            return None, []
        
        print(f"✓ Processing image: {image_path}")
        
        # Process frame
        processed_image, detections = self.process_frame(image)
        
        # Save output if specified
        if output_path:
            cv2.imwrite(output_path, processed_image)
            print(f"✓ Output saved to: {output_path}")
        
        # Display result
        cv2.imshow('Traffic Sign Recognition', processed_image)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return processed_image, detections


def main():
    """Main function to demonstrate the system"""
    
    # Initialize the recognition system
    system = TrafficSignRecognitionSystem()
    
    if system.model is None:
        print("✗ Cannot proceed without a loaded model")
        return
    
    # Example 1: Process camera stream
    print("\n" + "="*60)
    print("TRAFFIC SIGN RECOGNITION SYSTEM")
    print("="*60)
    print("\nChoose mode:")
    print("1. Camera stream (real-time)")
    print("2. Video file")
    print("3. Image file")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        print("\nStarting camera stream...")
        print("Press 'q' to quit")
        system.process_video(
            video_source=0,
            output_path=None,  # Set to a file path to save output
            confidence_threshold=0.5
        )
    
    elif choice == '2':
        video_path = input("Enter video file path: ").strip()
        output_path = input("Enter output file path (or press Enter to skip): ").strip()
        if not output_path:
            output_path = None
        
        print("\nProcessing video...")
        print("Press 'q' to quit")
        system.process_video(
            video_source=video_path,
            output_path=output_path,
            confidence_threshold=0.5
        )
    
    elif choice == '3':
        image_path = input("Enter image file path: ").strip()
        output_path = input("Enter output file path (or press Enter to skip): ").strip()
        if not output_path:
            output_path = None
        
        print("\nProcessing image...")
        system.process_image(
            image_path=image_path,
            output_path=output_path
        )
    
    elif choice == '4':
        print("Exiting...")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()

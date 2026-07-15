import pygame
import sys
import queue
import time
import numpy as np
import subprocess
import json
import json
import os
import re
import pytesseract
from PIL import ImageGrab
from core.predictor import PredictorThread

# Configuration for Screen Capture
# You must adjust these to match the bounding box of the skribbl.io canvas on your monitor!
CANVAS_X = 100
CANVAS_Y = 100
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600

TIMER_INTERVAL = 1.5 # seconds

def main():
    global CANVAS_X, CANVAS_Y, CANVAS_WIDTH, CANVAS_HEIGHT
    pygame.init()
    screen = pygame.display.set_mode((350, 600))
    pygame.display.set_caption("Skribbl AI Companion")
    font = pygame.font.SysFont('Arial', 18)
    large_font = pygame.font.SysFont('Arial', 24, bold=True)
    clock = pygame.time.Clock()
    
    # Predictor
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    predictions = []
    past_guesses = []
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(base_dir, 'data', 'vocab.txt')
    
    predictor = PredictorThread(vocab_path, input_queue, output_queue)
    predictor.start()
    
    # UI Elements
    target_length = 5
    btn_minus_rect = pygame.Rect(20, 20, 40, 40)
    btn_plus_rect = pygame.Rect(140, 20, 40, 40)
    btn_clear_rect = pygame.Rect(250, 20, 80, 40)
    btn_box_rect = pygame.Rect(20, 70, 110, 40)
    btn_hint_box_rect = pygame.Rect(140, 70, 110, 40)
    
    last_grab_time = time.time()
    bbox_process = None
    hint_bbox_process = None
    current_hint = ""
    
    while True:
        current_time = time.time()
        
        # Screen capture timer
        if current_time - last_grab_time >= TIMER_INTERVAL:
            last_grab_time = current_time
            
            # Read updated bbox
            bbox_path = os.path.join(base_dir, 'data', 'canvas_bbox.json')
            if os.path.exists(bbox_path):
                try:
                    with open(bbox_path, "r") as f:
                        data = json.load(f)
                        CANVAS_X = data["x"]
                        CANVAS_Y = data["y"]
                        CANVAS_WIDTH = data["width"]
                        CANVAS_HEIGHT = data["height"]
                except Exception as e:
                    pass

            bbox = (CANVAS_X, CANVAS_Y, CANVAS_X + CANVAS_WIDTH, CANVAS_Y + CANVAS_HEIGHT)
            try:
                img = ImageGrab.grab(bbox)
                
                # Preprocessing to remove colors and normalize to black strokes on white background
                gray_img = img.convert('L')
                gray_array = np.array(gray_img)
                
                # Threshold: Anything darker than 240 becomes black (stroke), else white (background)
                thresholded = np.where(gray_array < 240, 0, 255).astype(np.uint8)
                
                # Convert back to 3-channel RGB for the ResNet model
                canvas_array = np.stack((thresholded,)*3, axis=-1)
                
                # Check for hint bbox
                hint_bbox_path = os.path.join(base_dir, 'data', 'hint_bbox.json')
                hint_pattern = target_length
                
                if os.path.exists(hint_bbox_path):
                    try:
                        with open(hint_bbox_path, "r") as f:
                            hdata = json.load(f)
                            hbbox = (hdata["x"], hdata["y"], hdata["x"] + hdata["width"], hdata["y"] + hdata["height"])
                        hint_img = ImageGrab.grab(hbbox)
                        
                        # OCR
                        ocr_text = pytesseract.image_to_string(hint_img)
                        cleaned_hint = ""
                        for char in ocr_text:
                            if char.isspace():
                                continue
                            if char.isalpha():
                                cleaned_hint += char.lower()
                            else:
                                cleaned_hint += "_"
                        if cleaned_hint:
                            hint_pattern = cleaned_hint
                    except Exception as e:
                        pass
                
                current_hint = hint_pattern
                input_queue.put((canvas_array, hint_pattern))
            except Exception as e:
                print(f"Error capturing screen: {e}")

        # Pygame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if bbox_process:
                    bbox_process.terminate()
                if hint_bbox_process:
                    hint_bbox_process.terminate()
                predictor.stop()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    pos = event.pos
                    if btn_minus_rect.collidepoint(pos):
                        target_length = max(1, target_length - 1)
                        last_grab_time = 0 # Force immediate grab on change
                    elif btn_plus_rect.collidepoint(pos):
                        target_length += 1
                        last_grab_time = 0 # Force immediate grab on change
                    elif btn_clear_rect.collidepoint(pos):
                        past_guesses = []
                    elif btn_box_rect.collidepoint(pos):
                        if bbox_process is None or bbox_process.poll() is not None:
                            bbox_script = os.path.join(base_dir, 'ui', 'boundary_box.py')
                            bbox_process = subprocess.Popen([sys.executable, bbox_script])
                        else:
                            bbox_process.terminate()
                            bbox_process = None
                    elif btn_hint_box_rect.collidepoint(pos):
                        if hint_bbox_process is None or hint_bbox_process.poll() is not None:
                            hint_script = os.path.join(base_dir, 'ui', 'hint_box.py')
                            hint_bbox_process = subprocess.Popen([sys.executable, hint_script])
                        else:
                            hint_bbox_process.terminate()
                            hint_bbox_process = None
        
        # Read AI Predictions
        while not output_queue.empty():
            try:
                predictions = output_queue.get_nowait()
                # Update past guesses based on accuracy
                past_dict = {w: p for w, p in past_guesses}
                for w, p in predictions:
                    if w not in past_dict or p > past_dict[w]:
                        past_dict[w] = p
                past_guesses = sorted(past_dict.items(), key=lambda x: x[1], reverse=True)
            except queue.Empty:
                break
                
        # Draw UI
        screen.fill((240, 240, 240))
        
        # Length controls
        pygame.draw.rect(screen, (200, 200, 200), btn_minus_rect)
        pygame.draw.rect(screen, (200, 200, 200), btn_plus_rect)
        pygame.draw.rect(screen, (200, 100, 100), btn_clear_rect)
        pygame.draw.rect(screen, (100, 150, 200), btn_box_rect)
        pygame.draw.rect(screen, (150, 100, 200), btn_hint_box_rect)
        
        screen.blit(font.render("-", True, (0,0,0)), (btn_minus_rect.centerx - 5, btn_minus_rect.centery - 10))
        screen.blit(font.render("+", True, (0,0,0)), (btn_plus_rect.centerx - 5, btn_plus_rect.centery - 10))
        screen.blit(font.render("Clear", True, (255,255,255)), (btn_clear_rect.x + 15, btn_clear_rect.y + 10))
        box_text = "Canvas Box" if (bbox_process is None or bbox_process.poll() is not None) else "Close Box"
        screen.blit(font.render(box_text, True, (255,255,255)), (btn_box_rect.x + 10, btn_box_rect.y + 10))
        
        hint_text = "Hint Box" if (hint_bbox_process is None or hint_bbox_process.poll() is not None) else "Close Hint"
        screen.blit(font.render(hint_text, True, (255,255,255)), (btn_hint_box_rect.x + 15, btn_hint_box_rect.y + 10))
        
        # Display current hint or length
        display_hint = str(current_hint) if current_hint else str(target_length)
        screen.blit(font.render(f"Hint: {display_hint}", True, (0,0,0)), (20, 120))
        
        pygame.draw.line(screen, (150, 150, 150), (0, 145), (350, 145), 2)
        
        # Current predictions
        screen.blit(large_font.render("Current Guesses:", True, (0,0,0)), (20, 150))
        for i, (word, prob) in enumerate(predictions):
            txt = f"{i+1}. {word} ({(prob*100):.1f}%)"
            screen.blit(font.render(txt, True, (0, 100, 0)), (20, 180 + i * 30))
            
        # Past predictions
        screen.blit(large_font.render("All-Time Highest:", True, (0,0,0)), (20, 280))
        for i, (word, prob) in enumerate(past_guesses[:10]):
            txt = f"{i+1}. {word} ({(prob*100):.1f}%)"
            screen.blit(font.render(txt, True, (100, 0, 0)), (20, 310 + i * 30))
            
        pygame.display.flip()
        clock.tick(30)

if __name__ == '__main__':
    main()

import pygame
import sys
import numpy as np
import queue
from ui_components import UI
from predictor import PredictorThread

WIDTH, HEIGHT = 1000, 700

def flood_fill(surface, pos, target_color, fill_color):
    """
    Custom flood fill implementation.
    """
    if target_color == fill_color:
        return
        
    width = surface.get_width()
    height = surface.get_height()
    
    # Needs to be a list for queue operations
    q = [pos]
    
    # We use PixelArray for faster pixel-level access in Pygame
    pixel_array = pygame.PixelArray(surface)
    
    target_int = surface.map_rgb(target_color)
    fill_int = surface.map_rgb(fill_color)
    
    # BFS
    visited = set()
    
    while q:
        x, y = q.pop(0)
        
        if (x, y) in visited:
            continue
            
        visited.add((x, y))
        
        if pixel_array[x, y] == target_int:
            pixel_array[x, y] = fill_int
            
            if x > 0: q.append((x - 1, y))
            if x < width - 1: q.append((x + 1, y))
            if y > 0: q.append((x, y - 1))
            if y < height - 1: q.append((x, y + 1))
            
    # Always delete or close pixel array to unlock the surface
    del pixel_array

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Skribbl.io AI Predictor Test")
    
    font = pygame.font.SysFont('Arial', 18)
    large_font = pygame.font.SysFont('Arial', 24, bold=True)
    
    ui = UI(WIDTH, HEIGHT)
    
    # Canvas surface (allows us to pass just the drawing to the AI)
    canvas = pygame.Surface(ui.canvas_rect.size)
    canvas.fill((255, 255, 255))
    
    # State
    drawing = False
    current_color = (0, 0, 0)
    current_tool = 'brush'
    current_size = 8
    last_pos = None
    
    # Predictor
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    predictions = []
    past_guesses = []
    
    predictor = PredictorThread('vocab.txt', input_queue, output_queue)
    predictor.start()
    
    clock = pygame.time.Clock()
    
    # Send initial blank frame
    canvas_array = pygame.surfarray.array3d(canvas)
    canvas_array = np.transpose(canvas_array, (1, 0, 2)) # H, W, C
    input_queue.put((canvas_array, ui.target_length))
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                predictor.stop()
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    pos = event.pos
                    
                    # Handle UI clicks
                    if ui.toolbar_rect.collidepoint(pos):
                        # Color buttons
                        for btn in ui.color_buttons:
                            if btn.is_clicked(pos):
                                current_color = btn.value
                                for b in ui.color_buttons: b.is_selected = False
                                btn.is_selected = True
                                
                        # Tool buttons
                        for btn in ui.tool_buttons:
                            if btn.is_clicked(pos):
                                current_tool = btn.value
                                for b in ui.tool_buttons: b.is_selected = False
                                btn.is_selected = True
                                if current_tool == 'clear':
                                    canvas.fill((255, 255, 255))
                                    current_tool = 'brush'
                                    ui.tool_buttons[0].is_selected = True
                                    past_guesses = [] # Reset historical guesses
                                    # Trigger AI
                                    canvas_array = pygame.surfarray.array3d(canvas)
                                    canvas_array = np.transpose(canvas_array, (1, 0, 2))
                                    input_queue.put((canvas_array, ui.target_length))
                                    
                        # Size buttons
                        for btn in ui.size_buttons:
                            if btn.is_clicked(pos):
                                current_size = btn.value
                                for b in ui.size_buttons: b.is_selected = False
                                btn.is_selected = True
                                
                        # Length buttons
                        if ui.length_btn_minus.is_clicked(pos):
                            ui.target_length = max(1, ui.target_length - 1)
                            # Trigger AI
                            canvas_array = pygame.surfarray.array3d(canvas)
                            canvas_array = np.transpose(canvas_array, (1, 0, 2))
                            input_queue.put((canvas_array, ui.target_length))
                            
                        if ui.length_btn_plus.is_clicked(pos):
                            ui.target_length += 1
                            # Trigger AI
                            canvas_array = pygame.surfarray.array3d(canvas)
                            canvas_array = np.transpose(canvas_array, (1, 0, 2))
                            input_queue.put((canvas_array, ui.target_length))
                            
                    elif ui.canvas_rect.collidepoint(pos):
                        rel_pos = (pos[0] - ui.canvas_rect.x, pos[1] - ui.canvas_rect.y)
                        if current_tool == 'brush':
                            drawing = True
                            last_pos = rel_pos
                            pygame.draw.circle(canvas, current_color, rel_pos, current_size // 2)
                        elif current_tool == 'fill':
                            target_color = canvas.get_at(rel_pos)[:3]
                            if target_color != current_color:
                                flood_fill(canvas, rel_pos, target_color, current_color)
                                # Trigger AI immediately after fill
                                canvas_array = pygame.surfarray.array3d(canvas)
                                canvas_array = np.transpose(canvas_array, (1, 0, 2))
                                input_queue.put((canvas_array, ui.target_length))
                                
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and drawing:
                    drawing = False
                    # Trigger AI at end of stroke
                    canvas_array = pygame.surfarray.array3d(canvas)
                    canvas_array = np.transpose(canvas_array, (1, 0, 2))
                    input_queue.put((canvas_array, ui.target_length))
                    
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    pos = event.pos
                    # Clamp to canvas
                    if ui.canvas_rect.collidepoint(pos):
                        rel_pos = (pos[0] - ui.canvas_rect.x, pos[1] - ui.canvas_rect.y)
                        if last_pos:
                            pygame.draw.line(canvas, current_color, last_pos, rel_pos, current_size)
                        last_pos = rel_pos

        # Read AI predictions
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
                
        # Draw everything
        screen.fill((255, 255, 255))
        screen.blit(canvas, ui.canvas_rect)
        ui.draw(screen, font, large_font, predictions, past_guesses)
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()

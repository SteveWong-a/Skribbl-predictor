import pygame

class Button:
    def __init__(self, x, y, width, height, color, text='', value=None, border_color=(0,0,0)):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.text = text
        self.value = value
        self.border_color = border_color
        self.is_selected = False
        
    def draw(self, surface, font):
        # Draw background
        pygame.draw.rect(surface, self.color, self.rect)
        
        # Draw border
        border_width = 3 if self.is_selected else 1
        pygame.draw.rect(surface, self.border_color, self.rect, border_width)
        
        # Draw text
        if self.text:
            text_surface = font.render(self.text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=self.rect.center)
            surface.blit(text_surface, text_rect)
            
    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

class UI:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Define layout areas
        self.toolbar_height = 80
        self.sidebar_width = 250
        
        self.canvas_rect = pygame.Rect(0, self.toolbar_height, width - self.sidebar_width, height - self.toolbar_height)
        self.toolbar_rect = pygame.Rect(0, 0, width, self.toolbar_height)
        self.sidebar_rect = pygame.Rect(width - self.sidebar_width, self.toolbar_height, self.sidebar_width, height - self.toolbar_height)
        
        # Colors
        self.colors = [
            (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255),
            (128, 128, 128), (139, 69, 19)
        ]
        
        self.setup_buttons()
        
    def setup_buttons(self):
        self.color_buttons = []
        start_x = 10
        y = 10
        btn_size = 30
        spacing = 10
        
        # Color palette
        for i, color in enumerate(self.colors):
            x = start_x + (i * (btn_size + spacing))
            btn = Button(x, y, btn_size, btn_size, color, value=color)
            if i == 0:
                btn.is_selected = True # Default black
            self.color_buttons.append(btn)
            
        # Tool buttons
        tools_start_x = start_x + (len(self.colors) * (btn_size + spacing)) + 30
        self.tool_buttons = [
            Button(tools_start_x, y, 60, 30, (200, 200, 200), text='Brush', value='brush'),
            Button(tools_start_x + 70, y, 60, 30, (200, 200, 200), text='Fill', value='fill'),
            Button(tools_start_x + 140, y, 60, 30, (200, 200, 200), text='Clear', value='clear')
        ]
        self.tool_buttons[0].is_selected = True # Default brush
        
        # Size buttons
        sizes_start_x = tools_start_x + 230
        self.size_buttons = [
            Button(sizes_start_x, y, 40, 30, (200, 200, 200), text='S', value=3),
            Button(sizes_start_x + 50, y, 40, 30, (200, 200, 200), text='M', value=8),
            Button(sizes_start_x + 100, y, 40, 30, (200, 200, 200), text='L', value=15)
        ]
        self.size_buttons[1].is_selected = True # Default medium
        
        # Target Length +/- buttons
        target_start_x = sizes_start_x + 170
        self.target_length = 5
        self.length_btn_minus = Button(target_start_x, y, 30, 30, (200, 200, 200), text='-')
        self.length_btn_plus = Button(target_start_x + 90, y, 30, 30, (200, 200, 200), text='+')
        self.target_label_rect = pygame.Rect(target_start_x + 35, y, 50, 30)

    def draw(self, surface, font, large_font, predictions, past_guesses=[]):
        # Draw panels
        pygame.draw.rect(surface, (230, 230, 230), self.toolbar_rect)
        pygame.draw.rect(surface, (240, 240, 240), self.sidebar_rect)
        
        # Draw borders
        pygame.draw.line(surface, (150, 150, 150), (0, self.toolbar_height), (self.width, self.toolbar_height), 2)
        pygame.draw.line(surface, (150, 150, 150), (self.width - self.sidebar_width, self.toolbar_height), (self.width - self.sidebar_width, self.height), 2)
        
        # Draw buttons
        for btn in self.color_buttons + self.tool_buttons + self.size_buttons:
            btn.draw(surface, font)
            
        self.length_btn_minus.draw(surface, font)
        self.length_btn_plus.draw(surface, font)
        
        # Target length label
        len_surf = font.render(f"Len: {self.target_length}", True, (0, 0, 0))
        len_rect = len_surf.get_rect(center=self.target_label_rect.center)
        surface.blit(len_surf, len_rect)
        
        # Draw current predictions sidebar
        title_surf = large_font.render("Current Guesses:", True, (0, 0, 0))
        surface.blit(title_surf, (self.sidebar_rect.x + 20, self.sidebar_rect.y + 20))
        
        for i, (word, prob) in enumerate(predictions):
            guess_text = f"{i+1}. {word} ({(prob*100):.1f}%)"
            guess_surf = font.render(guess_text, True, (0, 100, 0))
            surface.blit(guess_surf, (self.sidebar_rect.x + 20, self.sidebar_rect.y + 50 + i * 30))
            
        # Draw past historical guesses
        hist_title = large_font.render("All-Time Highest:", True, (0, 0, 0))
        surface.blit(hist_title, (self.sidebar_rect.x + 20, self.sidebar_rect.y + 160))
        
        for i, (word, prob) in enumerate(past_guesses[:10]):
            hist_text = f"{i+1}. {word} ({(prob*100):.1f}%)"
            hist_surf = font.render(hist_text, True, (100, 0, 0))
            surface.blit(hist_surf, (self.sidebar_rect.x + 20, self.sidebar_rect.y + 190 + i * 30))

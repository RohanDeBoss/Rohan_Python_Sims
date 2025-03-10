import pygame
import random
import sys
import os

# ---------------------
# Constants & Colors
# ---------------------
WIN_WIDTH, WIN_HEIGHT = 800, 600
FPS = 60

WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
RED    = (255, 0, 0)
GREEN  = (0, 255, 0)
# Lighter blue for better visibility
BLUE   = (30, 144, 255)
YELLOW = (255, 255, 0)
GRAY   = (50, 50, 50)
PURPLE = (160, 32, 240)

# ---------------------
# Game States
# ---------------------
STATE_MENU = "menu"
STATE_PLAYING = "playing"
STATE_GAME_OVER = "game_over"

# ---------------------
# Custom Events
# ---------------------
BLOCK_SPAWN_EVENT = pygame.USEREVENT + 1
POWERUP_SPAWN_EVENT = pygame.USEREVENT + 2

# ---------------------
# Player Class using Sprites
# ---------------------
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.width = 50
        self.height = 50
        self.speed = 9
        # Draw a smooth green circle
        self.image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.ellipse(self.image, GREEN, (0, 0, self.width, self.height))
        self.rect = self.image.get_rect(midbottom=(WIN_WIDTH // 2, WIN_HEIGHT - 20))
    
    def update(self, keys=None, *args, **kwargs):
        if keys is None:
            keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] and self.rect.right < WIN_WIDTH:
            self.rect.x += self.speed

# ---------------------
# Block (Obstacle) Class using Sprites
# ---------------------
class Block(pygame.sprite.Sprite):
    def __init__(self, speed):
        super().__init__()
        self.size = random.randint(30, 70)
        self.speed = speed
        self.color = random.choice([RED, YELLOW, BLUE])
        # Draw a rounded rectangle
        self.image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.rect(self.image, self.color, (0, 0, self.size, self.size), border_radius=8)
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, WIN_WIDTH - self.size)
        self.rect.y = -self.size

    def update(self, *args, **kwargs):
        self.rect.y += self.speed
        # Note: We removed the automatic kill here.
        
# ---------------------
# Power-Up Class using Sprites
# ---------------------
class PowerUp(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.size = 30
        # Draw a circle for the power-up
        self.image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.circle(self.image, PURPLE, (self.size // 2, self.size // 2), self.size // 2)
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, WIN_WIDTH - self.size)
        self.rect.y = -self.size
        self.speed = 3

    def update(self, *args, **kwargs):
        self.rect.y += self.speed
        if self.rect.top > WIN_HEIGHT:
            self.kill()

# ---------------------
# Main Game Class
# ---------------------
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption("Dodge the Blocks!")
        self.clock = pygame.time.Clock()
        self.state = STATE_MENU
        
        # High score management
        self.high_score_file = "FB_highscore.txt"
        self.high_score = self.load_high_score()

        # Timer events for block and powerup spawns
        self.base_spawn_delay = 800  # in milliseconds (starting delay)
        self.base_powerup_delay = 10000  # in milliseconds
        pygame.time.set_timer(BLOCK_SPAWN_EVENT, self.base_spawn_delay)
        pygame.time.set_timer(POWERUP_SPAWN_EVENT, self.base_powerup_delay)

        # Create a pre-rendered gradient background
        self.background = self.create_background()

    def create_background(self):
        background = pygame.Surface((WIN_WIDTH, WIN_HEIGHT))
        # Gradient from dark purple to near-black
        for y in range(WIN_HEIGHT):
            r = int(50 + (y / WIN_HEIGHT) * 50)  # 50 to 100
            g = 0
            b = int(50 + (y / WIN_HEIGHT) * 205)  # 50 to 255
            pygame.draw.line(background, (r, g, b), (0, y), (WIN_WIDTH, y))
        return background

    def load_high_score(self):
        if os.path.exists(self.high_score_file):
            try:
                with open(self.high_score_file, "r") as f:
                    return int(f.read().strip())
            except:
                return 0
        return 0

    def save_high_score(self):
        with open(self.high_score_file, "w") as f:
            f.write(str(self.high_score))

    def draw_text(self, text, size, color, x, y, center=True):
        font = pygame.font.SysFont(None, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        self.screen.blit(text_surface, text_rect)

    # ---------------------
    # Menu Screen
    # ---------------------
    def menu(self):
        while self.state == STATE_MENU:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit(); sys.exit()
                    else:
                        self.state = STATE_PLAYING
            
            self.screen.blit(self.background, (0, 0))
            self.draw_text("DODGE THE BLOCKS!", 72, YELLOW, WIN_WIDTH // 2, WIN_HEIGHT // 4)
            self.draw_text("Use LEFT/RIGHT arrows to move", 32, WHITE, WIN_WIDTH // 2, WIN_HEIGHT // 2 - 40)
            self.draw_text("Catch the purple power-ups for slow-mo!", 28, WHITE, WIN_WIDTH // 2, WIN_HEIGHT // 2)
            self.draw_text("Press any key to start, Q to Quit", 28, WHITE, WIN_WIDTH // 2, WIN_HEIGHT // 2 + 60)
            self.draw_text(f"High Score: {self.high_score}", 32, WHITE, WIN_WIDTH // 2, WIN_HEIGHT - 60)
            pygame.display.flip()

    # ---------------------
    # Main Game Play
    # ---------------------
    def play(self):
        all_sprites = pygame.sprite.Group()
        block_group = pygame.sprite.Group()
        powerup_group = pygame.sprite.Group()

        player = Player()
        all_sprites.add(player)
        
        score = 0
        # Starting block speed (increases as score increases)
        block_speed = 2  
        # Spawn delay variable (in ms); will gradually decrease to a minimum value (max frequency)
        spawn_delay = self.base_spawn_delay  
        powerup_active = False
        powerup_timer = 0
        POWERUP_DURATION = 4000  # slow-mo lasts 4 seconds

        # Set the initial timer for block spawning
        pygame.time.set_timer(BLOCK_SPAWN_EVENT, spawn_delay)

        while running := (self.state == STATE_PLAYING):
            dt = self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == BLOCK_SPAWN_EVENT:
                    block = Block(block_speed)
                    all_sprites.add(block)
                    block_group.add(block)
                if event.type == POWERUP_SPAWN_EVENT:
                    powerup = PowerUp()
                    all_sprites.add(powerup)
                    powerup_group.add(powerup)
            
            keys = pygame.key.get_pressed()
            all_sprites.update(keys)

            if pygame.sprite.spritecollide(player, block_group, False):
                self.state = STATE_GAME_OVER
                running = False

            hits = pygame.sprite.spritecollide(player, powerup_group, True)
            if hits:
                powerup_active = True
                powerup_timer = pygame.time.get_ticks()
                # Immediately reduce new block speed by 2 when slow-mo is activated
                block_speed = max(2, block_speed - 2)
            
            # End slow-mo after 4 seconds and restore block speed based on score
            if powerup_active and pygame.time.get_ticks() - powerup_timer > POWERUP_DURATION:
                powerup_active = False
                block_speed = 2 + score // 5
                block_speed = min(block_speed, 10)
            
            # Check for blocks that have moved off screen, update score and adjust speed
            for block in block_group.copy():
                if block.rect.top > WIN_HEIGHT:
                    block.kill()
                    score += 1
                    if not powerup_active:
                        block_speed = 2 + score // 5
                        block_speed = min(block_speed, 15)

            # Gradually increase spawn frequency (i.e. reduce delay) until a minimum of 200 ms
            new_delay = max(200, 800 - 30 * (score // 5))
            if new_delay != spawn_delay:
                spawn_delay = new_delay
                pygame.time.set_timer(BLOCK_SPAWN_EVENT, spawn_delay)

            # Draw background, sprites, and HUD info
            self.screen.blit(self.background, (0, 0))
            all_sprites.draw(self.screen)
            self.draw_text(f"Score: {score}   Speed: {block_speed}   Spawn Delay: {spawn_delay} ms", 
                           30, WHITE, 10, 10, center=False)
            
            if powerup_active:
                overlay = pygame.Surface((WIN_WIDTH, WIN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((128, 0, 128, 100))
                self.screen.blit(overlay, (0, 0))
                self.draw_text("SLOW-MO ACTIVE!", 36, YELLOW, WIN_WIDTH // 2, 30)

            pygame.display.flip()

        return score

    # ---------------------
    # Game Over Screen
    # ---------------------
    def game_over(self, score):
        if score > self.high_score:
            self.high_score = score
            self.save_high_score()
        
        while self.state == STATE_GAME_OVER:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    self.state = STATE_MENU
            
            self.screen.blit(self.background, (0, 0))
            self.draw_text("GAME OVER", 72, RED, WIN_WIDTH // 2, WIN_HEIGHT // 3)
            self.draw_text(f"Score: {score}", 36, WHITE, WIN_WIDTH // 2, WIN_HEIGHT // 2)
            self.draw_text(f"High Score: {self.high_score}", 36, WHITE, WIN_WIDTH // 2, WIN_HEIGHT // 2 + 50)
            self.draw_text("Press any key to return to menu", 28, WHITE, WIN_WIDTH // 2, WIN_HEIGHT // 2 + 100)
            pygame.display.flip()

    def run(self):
        while True:
            if self.state == STATE_MENU:
                self.menu()
            elif self.state == STATE_PLAYING:
                final_score = self.play()
            elif self.state == STATE_GAME_OVER:
                self.game_over(final_score)

if __name__ == "__main__":
    game = Game()
    game.run()

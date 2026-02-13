import pygame
import random

pygame.init()

# --- Window ---
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rock Paper Scissors")

# --- Font ---
font = pygame.font.Font(None, 36)

# --- Game data ---
options = ["rock", "paper", "scissor"]
rules = {"rock": "scissor", "paper": "rock", "scissor": "paper"}

# --- Button positions ---
buttons = {
    "rock": pygame.Rect(50, 300, 150, 50),
    "paper": pygame.Rect(225, 300, 150, 50),
    "scissor": pygame.Rect(400, 300, 150, 50)
}

# --- Scores ---
player_score = 0
computer_score = 0
winner_text = ""

# --- Game loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            for choice, rect in buttons.items():
                if rect.collidepoint(pos):
                    player_choice = choice
                    computer_choice = random.choice(options)

                    # Determine winner
                    if player_choice == computer_choice:
                        winner_text = f"DRAW! Both chose {player_choice}"
                    elif rules[player_choice] == computer_choice:
                        winner_text = f"YOU WON! {player_choice} beats {computer_choice}"
                        player_score += 1
                    else:
                        winner_text = f"YOU LOST! {computer_choice} beats {player_choice}"
                        computer_score += 1

    # --- Drawing ---
    screen.fill((255, 255, 255))  # clear screen

    # Draw buttons
    for choice, rect in buttons.items():
        pygame.draw.rect(screen, (0, 200, 0), rect)
        text = font.render(choice.capitalize(), True, (255, 255, 255))
        screen.blit(text, (rect.x + 20, rect.y + 10))

    # Draw winner text
    text = font.render(winner_text, True, (0, 0, 0))
    screen.blit(text, (50, 50))

    # Draw scores
    score_text = font.render(f"Player: {player_score}  Computer: {computer_score}", True, (0, 0, 0))
    screen.blit(score_text, (50, 100))

    pygame.display.flip()

pygame.quit()

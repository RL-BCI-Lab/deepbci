import pygame
import time


def get_time():
    return time.monotonic()
 
def get_action():
    for event in pygame.event.get():
        # Allows game to end correctly
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # Listens for mouse button
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return 1
    return 0
    
def run():
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((420, 420), pygame.DOUBLEBUF | pygame.HWSURFACE)

    screen.fill((255,255,255))
    screen.set_alpha(None)
    pygame.display.update()
    start_time = pygame.time.get_ticks()
    frame = 0
    min_time = 5
    max_time = 1
    atype = False
    clock = pygame.time.Clock()

    while True:
        dt = clock.tick(60)
        frame += 1
        action = get_action()
        
        if action == 1:
            atype = True
            action_start_time = pygame.time.get_ticks()
            action_start_frame = frame
            time.sleep(2)

        if atype and (pygame.time.get_ticks() - action_start_time) / 1000 >= max_time:
             atype = False 
             
        print("Frame: {} Time: {:.5f} Action: {}".format(frame, 
            (pygame.time.get_ticks() - start_time) / 1000, atype))
run()
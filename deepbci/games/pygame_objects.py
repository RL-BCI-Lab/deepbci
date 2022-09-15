import pygame

class GameScreen():
    """ Define and set pygame related screen variables.

        Attributes:
            height (int): height of screen

            width (int): width of screen

        Methods:
            screen (pygame.Screen): Pygame Surface Object

    """
    def __init__(self, width, height):
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode(
            (width, height), pygame.DOUBLEBUF | pygame.HWSURFACE)
    
        self.screen.fill((255,255,255))
        self.screen.set_alpha(None)
        pygame.display.update()

    def get_screen(self):
        """ Get a copy of the pygame.Screen.

            Returns:
                Returns a copy of the current pygame.Screen.
        """
        return self.screen.copy()

class Circle():
    def __init__(self, screen, radius, screen_width, screen_height):
        self.screen_width = screen_width 
        self.screen_height = screen_height
        self.screen = screen
        self.radius = radius
        self.x = screen_width // 2
        self.y = screen_height // 2
        
    def coord(self):
        return (self.circle.centerx, self.circle.centery)

    def resize_screen(self, width, height):
        self.screen_height = height
        self.screen_width = width

    def draw(self, color):
        self.circle = pygame.draw.circle(self.screen,
                                        color,
                                        (self.x,self.y),
                                        self.radius)
                                        
class Block():
    """ 
        Base class for drawing blocks via pygames.

        Args:

            width (int): width of block

            height (int): height of block

            x (int): x coord for center of the screen

            y (int): y coord for center of the screen

            rect (pygame.Rect): contains information for drawing a rectangle

    """
    def __init__(self, screen, width, height, x_offset):
        self.screen = screen
        self.width = width
        self.height = height
        self.x, self.y = [dim//2 for dim in screen.get_size()]
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.rect.center = (self.x + x_offset, self.y)

    def coord(self):
        """ 
            Gets the center coordinates of the block

            Return:
                (tuple): center (x, y) coordinates of block
        """
        return (self.rect.centerx, self.rect.centery)

    def centers_aligned(self, rect):
        """ 
            Checks if center of block is aligned with another pygame.Rect.

            Return:
                (bool): True if aligned, False if not.
        """
        if (self.rect.centerx == rect.centerx 
            and self.rect.centery == rect.centery):
            
            return True

        return False

    def draw(self, color):
        """Draws the pygame.Rect representing the block.

            Note:
                Will not actually be drawn on screen without the use of
                pygame.display.update().

            Args:
                color (tuple): A tuple of ints representing RGB
        """
        self.rect = pygame.draw.rect(self.screen, color, self.rect)
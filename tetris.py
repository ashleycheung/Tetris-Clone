"""
This is a python implementation of tetris
It is implemented using MVC architecture
"""
import pygame
import math
import random
from PIL import Image

#Grid
GRID_HEIGHT = 22
GRID_WIDTH = 10
CELL_SIZE = 40
CELL_MARGIN_SIZE = 1
FRAME_RATE = 60
BORDER_SIZE = 100

#Score
SCORE_FONT_SIZE = 30
FONT_COLOUR = (255,255,255)
SCORE_POS = (10,10)

#Miscellanious
BG_COLOUR = (20,20,20)
GRID_LINE_COLOUR = (10,10,10)
EMPTY_CELL_COLOUR = (0,0,0)
#How long in seconds before the block falls
FALL_SPEED = 0.3

#Code for empty
EMPTY = 0

L_BLOCK = {
    'code' : 1,
    'colour' : (239,122,40),
    'matrix' : [
        [0,1,0],
        [0,1,0],
        [0,1,1]
    ]
}

J_BLOCK = {
    'code' : 2,
    'colour' : (88,102,175),
    'matrix' : [
        [0,1,0],
        [0,1,0],
        [1,1,0]
    ]
}

I_BLOCK = {
    'code' : 3,
    'colour' : (59,199,237),
    'matrix' : [
        [0,1,0,0],
        [0,1,0,0],
        [0,1,0,0],
        [0,1,0,0]
    ]
}

O_BLOCK = {
    'code' : 4,
    'colour' : (247,212,55),
    'matrix' : [
        [1,1],
        [1,1]
    ]
}

S_BLOCK = {
    'code' : 5,
    'colour' : (89,184,63),
    'matrix' : [
        [0,0,0],
        [0,1,1],
        [1,1,0]
    ]
}

T_BLOCK = {
    'code' : 6,
    'colour' : (176,78,158),
    'matrix' : [
        [1,1,1],
        [0,1,0],
        [0,0,0]
    ]
}

Z_BLOCK = {
    'code' : 7,
    'colour' : (240,68,45),
    'matrix' : [
        [1,1,0],
        [0,1,1],
        [0,0,0]
    ]
}

BLOCKS = [L_BLOCK, J_BLOCK, I_BLOCK, O_BLOCK, S_BLOCK, T_BLOCK, Z_BLOCK]

"""Given a code return the block"""
def get_block_by_code(code):
    for block in BLOCKS:
        if block['code'] == code:
            return block
    return None

"""Generate the colour map for the blocks"""
def make_colour_map():
    cmap = {}
    for block in BLOCKS:
        cmap[block['code']] = block['colour']
    return cmap

#Stores the mapping of code to colours
COLOUR_MAP = make_colour_map()


class TetrisShape:
    """Shape is a 2D array"""
    def __init__(self, shape):
        self.matrix = shape['matrix']
        self.colour = shape['colour']
        self.code = shape['code']
        self.size = len(shape['matrix'])

    """Rotates the shape clockwise"""
    def rotate(self, is_reversed = False):
        if not is_reversed:
            self.matrix = list(zip(*self.matrix[::-1]))
        else:
            self.matrix = list(zip(*self.matrix))[::-1]

    """
    Prints the current shape into the terminal.
    Used mainly for debugging
    """
    def print(self):
        for row in self.matrix:
            print(row)


"""Tetris Game class"""
class TetrisGame:
    def __init__(self, level, controller, headless = False, start_state = None):
        #Initialise pygame
        pygame.init()
        self.running = True
        self.headless = headless
        #Create model
        self.model = TetrisModel(GRID_WIDTH, GRID_HEIGHT, level)

        #Create and attach controller
        self.controller = controller
        self.controller.connect(self)

        #Run pygame view
        self.view = TetrisView(self)
        self.gameclock = pygame.time.Clock()
        self.level = level
        #Load state if given one
        if not start_state is None:
            self.load(start_state)

    """Reset the game"""
    def reset(self):
        self.model = TetrisModel(GRID_WIDTH, GRID_HEIGHT, self.level)

    """Runs the next step for the game"""
    def step(self):
        #Get delta
        delta = float(self.gameclock.tick(FRAME_RATE)) / 1000
        self.controller.listen_for_event()
        self.model.next_state(delta)
        self.view.render(self.model)

    """Plays the game normally"""
    def play(self):
        while self.running:
            self.step()

    """Load a specific starting state"""
    def load(self, state):
        self.model.load(state)

    """Returns a pillow image of the tetris board"""
    def fetch_frame(self):
        return self.view.fetch_frame()

    """
    Returns the heuristics of the current tetris game
    These include:
        Aggregate Height: Sum of the highest block of each row
        Holes: Defines as any empty space with at least one tile above it
        Bumpiness: Sum of the absolute differences between all pairs of adjacent columns
    """
    def get_heuristics(self):
        #Get grid
        grid = self.model.get_placed_grid()
        height = self.model.height
        width = self.model.width
        column_heights = {}

        #get number of holes
        holes = 0

        #Loop from top down left to right
        for y in range(height):
            for x in range(width):
                #If current cell is not empty and it is the first
                #block to be found in the column when coming from top down
                #(This would mean it is the highest block) then store it
                if grid[y][x] != EMPTY and not x in column_heights:
                    column_heights[x] = height - y
                #Check if it is a hole
                #You cant have a hole on the top column so ignore it
                elif grid[y][x] == EMPTY and y != 0:
                    if grid[y-1][x] != 0:
                        holes += 1

            #If the highest block in each column has been found, then break
            if len(column_heights) == width:
                break

        #Get aggregate height
        aggregate_height = 0
        #Columns with zero blocks will not be in the dictionary
        #But it is okay to skip them since they have a height of 0
        for x in column_heights.values():
            aggregate_height += x

        #Fill remaining columns with 0's
        for x in range(width):
            if not x in column_heights:
                column_heights[x] = 0

        #Calculate bumpiness
        bumpiness = 0
        for x in range(width):
            if x != 0:
                bumpiness += abs(column_heights[x] - column_heights[x-1])

        return aggregate_height, holes, bumpiness


"""View of the Game"""
class TetrisView:
    def __init__(self, game):
        self.game = game
        self.screen = pygame.display.set_mode(self.get_window_size())
        pygame.display.set_caption("Tetris")
        self.grid_pos = (BORDER_SIZE, BORDER_SIZE)


    """Fetches the frame of the game"""
    def fetch_frame(self):
        #Extract the grid from the whole game
        grid_rect = pygame.Rect(self.grid_pos[0], self.grid_pos[1] + 2 * CELL_SIZE, 
            CELL_SIZE * GRID_WIDTH, CELL_SIZE * (GRID_HEIGHT - 2))
        grid_surface = self.screen.subsurface(grid_rect)

        #Convert grid surface into bytes string
        str_format = 'RGBA'
        flipped = False
        raw_str = pygame.image.tostring(grid_surface, str_format, flipped)

        #Convert to Pillow image
        image = Image.frombytes(str_format, grid_surface.get_size(), raw_str)
        return image

    """Gets window size"""
    def get_window_size(self):
        return (CELL_SIZE * GRID_WIDTH + 2 * BORDER_SIZE, 
            CELL_SIZE * GRID_HEIGHT + 2* BORDER_SIZE)

    """Renders the view"""
    def render(self, model):
        #Draw grid
        self.render_background()
        self.draw_grid(model.get_grid())
        self.render_score(model)
        #Display on pygame window
        pygame.display.flip()


    """Renders background and grid base"""
    def render_background(self):
        self.screen.fill(BG_COLOUR)
        #Draw gridlines base
        #Subtract 2 because the first 2 rows dont count
        gridlines_base = pygame.Surface((CELL_SIZE * GRID_WIDTH, CELL_SIZE * (GRID_HEIGHT - 2)))
        gridlines_base.fill(GRID_LINE_COLOUR)
        self.screen.blit(gridlines_base, (self.grid_pos[0], self.grid_pos[1] + CELL_SIZE * 2))

    """Renders the score onto the game"""
    def render_score(self, model):
        font = pygame.font.SysFont('arial', SCORE_FONT_SIZE)
        text_surface = font.render(f"Level: {str(model.level)} Score: {str(model.score)}", True, FONT_COLOUR)
        self.screen.blit(text_surface, SCORE_POS)


    """Renders a cell"""
    def render_cell(self, cellstate, xpos, ypos):
        #Make a new surface
        new_cell = pygame.Surface((CELL_SIZE - 2 * CELL_MARGIN_SIZE, 
            CELL_SIZE - 2 * CELL_MARGIN_SIZE))

        #Fill cell
        if cellstate == EMPTY:
            if ypos == 0 or ypos == 1:
                new_cell.fill(BG_COLOUR)
            else:
                new_cell.fill(EMPTY_CELL_COLOUR)
        else:
            new_cell.fill(COLOUR_MAP[cellstate])

        #Position cell
        cellpos = (self.grid_pos[0] + xpos * CELL_SIZE + CELL_MARGIN_SIZE, 
            self.grid_pos[1] + ypos * CELL_SIZE + CELL_MARGIN_SIZE)
        self.screen.blit(new_cell, cellpos)


    """
    Draws the grid
    Goes from Left to right
    Top down
    """
    def draw_grid(self, grid):
        #Draw cells
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                self.render_cell(grid[i][j], j, i)


"""User Controller of the game"""
class TetrisController:
    def __init__(self):
        self.game = None

    """Connects to game"""
    def connect(self, game):
        self.game = game

    """Handle pygame events"""
    def handle_event(self, event):
        #Handle quit
        if event.type == pygame.QUIT:
            self.game.running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.game.model.go_left()
            elif event.key == pygame.K_RIGHT:
                self.game.model.go_right()
            elif event.key == pygame.K_DOWN:
                self.game.model.go_down()
            elif event.key == pygame.K_UP:
                self.game.model.rotate()
            elif event.key == pygame.K_SPACE:
                self.game.model.drop_block()

    """Thread that listens for events"""
    def listen_for_event(self):
        for event in pygame.event.get():
            self.handle_event(event)


"""Special controller used for bots"""
class BotController(TetrisController):
    def __init__(self):
        #Buffer to store actions
        self.action_buffer = []

    """Pass in the action as a string"""
    def action(self, action):
        self.action_buffer.append(action)

    """
    Overrides
    Event is now a string
    """
    def handle_event(self, event):
        if event == 'left':
            self.game.model.go_left()
        elif event == 'right':
            self.game.model.go_right()
        elif event == 'down':
            self.game.model.go_down()
        elif event == 'rotate':
            self.game.model.rotate()
        elif event == 'drop':
            self.game.model.drop_block()

    """
    Overrides
    Listens to the action buffer instead
    """
    def listen_for_event(self):
        #Handle action
        while self.action_buffer:
            self.handle_event(self.action_buffer.pop(0))

"""Model for the Tetris Game"""
class TetrisModel:
    def __init__(self, width, height, level):
        self.width = width
        self.height = height
        self.grid = self.make_grid()
        
        #Stores whether the game is over
        self.gameover = False

        #Player score
        self.score = 0
        self.lines_cleared = 0
        self.level = level

        #Stores information about the current shape
        self.current_shape = None
        self.shape_pos = None
        self.new_shape()

        #Stores all the timers
        self.timers = []

        #Add moving down timer
        self.timers.append(Timer(FALL_SPEED, self.go_down, loop=True))

    """
    Runs the next state
    Delta is time passed in seconds since last call
    """
    def next_state(self, delta):
        if not self.gameover:
            self.tick_timers(delta)

    """Ticks all the timers"""
    def tick_timers(self, delta):
        expired_timers = []
        #Tick every timer
        for timer in self.timers:
            timer.tick(delta)
            #Store expired timers
            if timer.finished:
                expired_timers.append(timer)
        
        #Remove all expired timers
        for timer in expired_timers:
            self.timers.remove(timer)

    """Loads a state"""
    def load(self, state):
        self.grid = state['grid']
        self.current_shape = TetrisShape(
            get_block_by_code(state['current_shape_code'])
        )
        self.shape_pos = self.get_spawn_pos(self.current_shape)

    """Returns whether the given position is valid on the grid"""
    def valid_pos(self, x, y):
        if y >= self.height or  y < 0 or x < 0 or x >= self.width:
            return False
        return True 

    """Returns the score for clearing the number of lines"""
    def calculate_score(self, lines_cleared):
        #Add to total lines cleared
        self.lines_cleared += lines_cleared
        if lines_cleared == 0:
            return 0
        elif lines_cleared == 1:
            return 40 * (self.level + 1)
        elif lines_cleared == 2:
            return 100 * (self.level + 1)
        elif lines_cleared == 3:
            return 300 * (self.level + 1)
        else:
            #4 lines cleared
            return 1200 * (self.level + 1)

    """Clears the lines in the game"""
    def clear_lines(self):
        num_lines_cleared = 0
        #Only need to check the lines which the shapes are part of
        for row_index in range(self.current_shape.size):
            line_cleared = True
            cell_pos_y = self.shape_pos[1] + row_index

            #Check that the row is in the grid
            if cell_pos_y >= self.height or cell_pos_y < 0:
                continue

            #Check that none of the cells in the row is empty
            for cell_pos_x in range(self.width):

                if not self.valid_pos(cell_pos_x, cell_pos_y):
                    continue

                if self.grid[cell_pos_y][cell_pos_x] == EMPTY:
                    line_cleared = False
                    break

            if line_cleared:
                self.print_grid()
                print(f"Row {cell_pos_y} cleared")
                #Remove the cleared row
                self.grid.pop(cell_pos_y)
                self.grid.insert(0,[EMPTY] * self.width)
                num_lines_cleared += 1
        
        self.score += self.calculate_score(num_lines_cleared)
        return num_lines_cleared

    """Checks if game over has occured yet"""
    def check_gameover(self):
        #Check if there is any block in the top row
        for cell in self.grid[0]:
            if cell != EMPTY:
                self.gameover = True
                print("Game Over!!")
                return True
        return False

    """Gets a new shape"""
    def new_shape(self):
        self.current_shape = self.next_shape()
        self.shape_pos = self.get_spawn_pos(self.current_shape)

    """Get the spawn position of a shape"""
    def get_spawn_pos(self, shape):
        return (int(self.width / 2 - math.ceil(float(shape.size)/2)), -1 * shape.size)

    """Makes next shape"""
    def next_shape(self):
        return TetrisShape(random.choice(BLOCKS))

    """
    Checks if a given shape at a given position is valid
    Position of the shape is its top left corner
    """
    def valid_position(self, shape, position):
        for y in range(shape.size):
            for x in range(shape.size):
                #Get grid coordinates
                grid_pos_x = position[0] + x
                grid_pos_y = position[1] + y

                #Check if the current cell is on the grid
                #Note: the position is still valid if the shape is above the grid
                #since it enters from above the grid
                in_grid = grid_pos_x >= 0 and grid_pos_x < self.width and grid_pos_y < self.height
                in_grid_view = in_grid and grid_pos_y >= 0

                #Check if the current cell on the shape is filled
                shape_cell_filled = shape.matrix[y][x] != EMPTY

                if in_grid_view:
                    #Check if the current grid cell is filled
                    grid_cell_filled = self.grid[grid_pos_y][grid_pos_x] != EMPTY
                    #Check if there is a collision
                    if grid_cell_filled and shape_cell_filled:
                        return False
                else:
                    #If shape goes out from the sides
                    if shape_cell_filled and not in_grid:
                        return False
        return True

    """
    Given a shape, replace everything it covers
    with the given code
    """
    def replace_shape(self, shape, pos, code, grid):
        for y in range(shape.size):
            for x in range(shape.size):
                if shape.matrix[y][x] != EMPTY:
                    #Get grid coordinates
                    grid_pos_x = pos[0] + x
                    grid_pos_y = pos[1] + y

                    #Check the cell is in the grid
                    in_grid_view = grid_pos_x >= 0 and grid_pos_x < self.width and grid_pos_y >= 0 and grid_pos_y < self.height
                    if in_grid_view:
                        grid[grid_pos_y][grid_pos_x] = code
        return grid

    """
    This assumes the shape is valid to draw
    Must check the new shape is valid first before running or error will occur
    """
    def draw_shape(self, shape, pos):
        self.grid = self.replace_shape(shape, pos, shape.code, self.grid)
    
    """
    Removes a shape in a given position
    Assumes the shape is in a valid position and the shape exists
    """
    def remove_shape(self, shape, pos):
        self.grid = self.replace_shape(shape, pos, EMPTY, self.grid)

    """
    Moves shape to the new position
    Returns whether the move was successful
    """
    def move_shape(self, new_pos):
        #Remove all the cells currently occupied by the shape
        self.remove_shape(self.current_shape, self.shape_pos)

        #Check if position is valid
        if not self.valid_position(self.current_shape, new_pos):
            #Roll back and re colour cell
            self.draw_shape(self.current_shape, self.shape_pos)
            return False

        #Add shape to grid
        self.draw_shape(self.current_shape, new_pos)
        self.shape_pos = new_pos
        return True
    
    """
    Sends the current shape down
    Returns whether the drop was successful
    """
    def go_down(self):
        #Check if the ground has been reached
        next_pos = (self.shape_pos[0], self.shape_pos[1] + 1)
        #If shape has reached the bottom make new shape
        if not self.move_shape(next_pos):
            self.clear_lines()
            self.check_gameover()
            self.new_shape()
            return False
        return True

    """Sends the shape left"""
    def go_left(self):
        #Check if the ground has been reached
        next_pos = (self.shape_pos[0] - 1, self.shape_pos[1])
        self.move_shape(next_pos)

    """Sends the shape right"""
    def go_right(self):
        #Check if the ground has been reached
        next_pos = (self.shape_pos[0] + 1, self.shape_pos[1])
        self.move_shape(next_pos)

    """Rotates the shape"""
    def rotate(self):
        #Remove all the cells currently occupied by the shape
        self.remove_shape(self.current_shape, self.shape_pos)
        #Rotate shape
        self.current_shape.rotate()
        #Check if rotation works
        if not self.valid_position(self.current_shape, self.shape_pos):
            #Roll back
            self.current_shape.rotate(is_reversed = True)
        
        #Draw shape again
        self.draw_shape(self.current_shape, self.shape_pos)
    
    """Drops the block to the bottom"""
    def drop_block(self):
        while self.go_down():
            pass

    """Makes grid of 0's"""
    def make_grid(self):
        output_grid = []
        for row in range(self.height):
            row = []
            for cell in range(self.width):
                row.append(EMPTY)
            output_grid.append(row)
        return output_grid

    """Returns the grid but only with the blocks that have been placed"""
    def get_placed_grid(self):
        return self.replace_shape(self.current_shape, self.shape_pos, EMPTY, self.grid)

    """Returns the current grid"""
    def get_grid(self):
        return self.grid

    """Prints grid into terminal. Used for debugging"""
    def print_grid(self, grid = None):
        if grid is None:
            grid = self.grid
        print("\n//////////Tetris Grid//////////")
        for row in grid:
            print(row)


"""A timer than runs a function after certain time delay in seconds"""
class Timer:
    def __init__(self, delay, function, loop = False):
        self.function = function
        self.time_elapsed = 0
        self.delay = delay
        #Stores whether timer is finished
        self.finished = False
        self.loop = loop

    """
    Ticks the timer for delta seconds
    """
    def tick(self, delta):
        self.time_elapsed += delta
        #If timer has finished run function
        if self.time_elapsed >= self.delay and not self.finished:
            self.function()

            if not self.loop:
                self.finished = True
            else:
                self.time_elapsed = 0



if __name__ == '__main__':
    t = TetrisGame(1, TetrisController())
    t.play()
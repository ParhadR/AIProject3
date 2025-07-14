import random

BLOCKED = "blocked_state"
OPEN = "open_state"

def generate_ship(D, p_factor):
    #D D grid with BLOCKED cells
    grid = [[BLOCKED for _ in range(D)] for _ in range(D)]

    # Pick a cell at random to start maze (but not on edge)
    def random_interior_cell():
        return random.randint(1, D - 2), random.randint(1, D - 2)

    starting_x, starting_y = random_interior_cell()
    grid[starting_x][starting_y] = OPEN  # open a starting cell

    # Count how many neighbors (in all 4 direction) are OPEN
    def count_open_neighbors(x, y):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return sum(
            1
            for dx, dy in directions
            if 0 <= x + dx < D and 0 <= y + dy < D and grid[x + dx][y + dy] == OPEN
        )

    #list of all current dead ends (open cells with 1 open neighbor)
    def find_dead_ends():
        dead_ends = []
        for x in range(1, D - 1):
            for y in range(1, D - 1):
                if grid[x][y] == OPEN and count_open_neighbors(x, y) == 1:
                    dead_ends.append((x, y))
        return dead_ends

    # Get blocked neighbor(s) (again, checking in all 4 direction) of a given cell
    def get_blocked_neighbors(x, y):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return [
            (x + dx, y + dy)
            for dx, dy in directions
            if 0 <= x + dx < D and 0 <= y + dy < D and grid[x + dx][y + dy] == BLOCKED
        ]

    # First, open cells to begin a maze-like structure
    while True:
        ship_maze = []

        # Look for blocked cells that are adjacent to exactly 1 open cell
        for x in range(1, D - 1):
            for y in range(1, D - 1):
                if grid[x][y] == BLOCKED and count_open_neighbors(x, y) == 1:
                    ship_maze.append((x, y))

        if not ship_maze:
            break  # if no more cells to open then initial maze complete

        # randomly pick one of the blocked cells next to an open cell and open that
        cell_to_open = random.choice(ship_maze)
        grid[cell_to_open[0]][cell_to_open[1]] = OPEN

    #Last, we wnat to reduce the amount of dead ends based on p_factor

    initial_dead_ends = find_dead_ends()
    target_dead_ends = int(len(initial_dead_ends) * (1 - p_factor))
    current_dead_ends = len(initial_dead_ends)

    while current_dead_ends > target_dead_ends and current_dead_ends > 0:
        #Pick a random dead end
        dead_end = random.choice(find_dead_ends())

        #Get any blocked neighbor that could be opened
        blocked_neighbors = get_blocked_neighbors(dead_end[0], dead_end[1])
        if blocked_neighbors:
            cell_to_open = random.choice(blocked_neighbors)
            grid[cell_to_open[0]][cell_to_open[1]] = OPEN

        # Update dead end count
        current_dead_ends = len(find_dead_ends())

    return grid  #Final maze/grid structure
"""
Pacsim NEAT implementation
"""

from __future__ import print_function
from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
import os
import neat
import visualize
import copy

class TotalPopulation(object):
    def __init__(self, config, config_ghost, initial_state=None):
        self.reporters = ReporterSet()
        self.reporters_g_1 = ReporterSet()
        self.reporters_g_2 = ReporterSet()

        self.config = config
        self.config_ghost = config_ghost

        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config, self.reporters, stagnation)
        self.reproduction_ghost = self.config_ghost.reproduction_type(config_ghost.reproduction_config, self.reporters_g_1, stagnation)

        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        else:
            raise Exception("Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.pacman_population = self.reproduction.create_new(config.genome_type, config.genome_config, config.pop_size)
            self.pacman_species = config.species_set_type(config, self.reporters)

            self.ghost_population_1 = self.reproduction_ghost.create_new(config_ghost.genome_type, config_ghost.genome_config, config_ghost.pop_size)
            self.ghost_population_2 = self.reproduction_ghost.create_new(config_ghost.genome_type, config_ghost.genome_config, config_ghost.pop_size)
            self.ghost_species_1 = config.species_set_type(config_ghost, self.reporters_g_1)
            self.ghost_species_2 = config.species_set_type(config_ghost, self.reporters_g_2)

            self.generation = 0
            self.pacman_species.speciate(config, self.pacman_population, self.generation)
            self.ghost_species_1.speciate(config_ghost, self.ghost_population_1, self.generation)
            self.ghost_species_2.speciate(config_ghost, self.ghost_population_2, self.generation)
        else:
            self.pacman_population, self.pacman_species, self.generation = initial_state
            self.ghost_population_1, self.ghost_species_1, self.generation = initial_state
            self.ghost_population_2, self.ghost_species_2, self.generation = initial_state

        self.best_genome_pacman = None
        self.best_genome_ghost_1 = None
        self.best_genome_ghost_2 = None

    def add_reporter_pacman(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter_pacman(self, reporter):
        self.reporters.remove(reporter)

    def add_reporter_ghost_1(self, reporter):
        self.reporters_g_1.add(reporter)

    def remove_reporter_ghost_1(self, reporter):
        self.reporters_g_1.remove(reporter)

    def add_reporter_ghost_2(self, reporter):
        self.reporters_g_2.add(reporter)

    def remove_reporter_ghost_2(self, reporter):
        self.reporters_g_2.remove(reporter)

    def run(self, evaluator, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.
        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.
        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.
        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.
        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """
        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)
            self.reporters_g_1.start_generation(self.generation)
            self.reporters_g_2.start_generation(self.generation)

            # Run the fitness evaluation function that will use all three populations
            evaluator(list(iteritems(self.pacman_population)), list(iteritems(self.ghost_population_1)), list(iteritems(self.ghost_population_2)), self.config, self.config_ghost)

            # Gather and report statistics.
            bestp = None
            for g in itervalues(self.pacman_population):
                if bestp is None or g.fitness > bestp.fitness:
                    bestp = g

            bestg1 = None
            for g in itervalues(self.ghost_population_1):
                if bestg1 is None or g.fitness > bestg1.fitness:
                    bestg1 = g

            bestg2 = None
            for g in itervalues(self.ghost_population_2):
                if not g.fitness:
                    continue
                if bestg2 is None or g.fitness > bestg2.fitness:
                    bestg2 = g

            self.reporters.post_evaluate(self.config, self.pacman_population, self.pacman_species, bestp)
            self.reporters_g_1.post_evaluate(self.config_ghost, self.ghost_population_1, self.ghost_species_1, bestg1)
            self.reporters_g_2.post_evaluate(self.config_ghost, self.ghost_population_2, self.ghost_species_2, bestg2)

            # Track the best genome ever seen.
            if self.best_genome_pacman is None or bestp.fitness > self.best_genome_pacman.fitness:
                self.best_genome_pacman = bestp

            if self.best_genome_ghost_1 is None or bestg1.fitness > self.best_genome_ghost_1.fitness:
                self.best_genome_ghost_1 = bestg1

            if self.best_genome_ghost_2 is None or bestg2.fitness > self.best_genome_ghost_2.fitness:
                self.best_genome_ghost_2 = bestg2

            # End if the fitness threshold is reached.
            fv = self.fitness_criterion(g.fitness for g in itervalues(self.pacman_population))
            if fv >= self.config.fitness_threshold:
                self.reporters.found_solution(self.config, self.generation, bestp)
                break

            # Create the next generation from the current generation.
            self.pacman_population = self.reproduction.reproduce(self.config, self.pacman_species,
                                                          self.config.pop_size, self.generation)
            self.ghost_population_1 = self.reproduction_ghost.reproduce(self.config_ghost, self.ghost_species_1,
                                                          self.config_ghost.pop_size, self.generation)
            self.ghost_population_2 = self.reproduction_ghost.reproduce(self.config_ghost, self.ghost_species_2,
                                              self.config_ghost.pop_size, self.generation)

            # Check for complete extinction.
            if not self.pacman_species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.pacman_population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            if not self.ghost_species_1.species:
                self.reporters_g_1.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config_ghost.reset_on_extinction:
                    self.ghost_population_1 = self.reproduction_ghost.create_new(self.config_ghost.genome_type,
                                                                   self.config_ghost.genome_config,
                                                                   self.config_ghost.pop_size)
                else:
                    raise CompleteExtinctionException()

            if not self.ghost_species_2.species:
                self.reporters_g_2.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config_ghost.reset_on_extinction:
                    self.ghost_population_2 = self.reproduction_ghost.create_new(self.config_ghost.genome_type,
                                                                   self.config_ghost.genome_config,
                                                                   self.config_ghost.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.pacman_species.speciate(self.config, self.pacman_population, self.generation)
            self.ghost_species_1.speciate(self.config_ghost, self.ghost_population_1, self.generation)
            self.ghost_species_2.speciate(self.config_ghost, self.ghost_population_2, self.generation)

            self.reporters.end_generation(self.config, self.pacman_population, self.pacman_species)
            self.reporters_g_1.end_generation(self.config_ghost, self.ghost_population_1, self.ghost_species_1)
            self.reporters_g_2.end_generation(self.config_ghost, self.ghost_population_2, self.ghost_species_2)

            self.generation += 1

        return self.best_genome_pacman, self.best_genome_ghost_1, self.best_genome_ghost_2


WINDOW_SIZE = 4

EMPTY = 0
FOOD = 1
FRUIT = 2
WALL = 3
PACMAN = 4
GHOST = 5

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

FOOD_SCORE = 10
WALL_SCORE = -25
MOVE_SCORE = 0
UNVISITED_SCORE = 25

PAC_STARTING_POS = (10, 5)
GHOSTONE = (0,0)
GHOSTTWO = (0,10)


PACMAN_BOARD = [
['G1', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'G2'] ,
['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*'] ,
['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*'] ,
['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*'] ,
['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*'] ,
['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*'] ,
['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*'] ,
['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*'] ,
['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*'] ,
['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*'] ,
['*', '*', '*', '*', '*', 'P', '*', '*', '*', '*', '*'] ,
]


translated_map = []
translate = {'#': WALL, '*': FOOD, 'G1': GHOST, ' ':EMPTY, 'P':PACMAN, 'G2': GHOST}
for row in PACMAN_BOARD:
    translated_map.append([])
    for cell in row:
        translated_map[-1].append(translate[cell])

PACMAN_BOARD = translated_map

test_map = [
    [3, 3, 3, 3, 3],
    [3, 1, 1, 1, 3],
    [3, 1, 3, 1, 3],
    [3, 1, 3, 1, 3],
    [3, 1, 4, 1, 3],
    [3, 3, 3, 3, 3]
]

dir_changes = [(1,0),(-1,-0),(0,1),(0,-1)]
wrap_on_edge = [(28,14),(-1,14)]
wrap_go_to = [(0,14),(27,14)]

def eval_genomes(genomes_pacman, genomes_ghost_1, genomes_ghost_2, config, config_ghost):
    """
    For all intents and purpacPoses, this is the fitness function
    Here, the pacsim will be used to evaluate each pacman agent
    Args:
        genomes: list of genomes from NEAT used to create the ANN
        config: config file with network parameters
    """
    for i in range(min(len(genomes_pacman), len(genomes_ghost_1))):
        genome_id_pacman, genome_pacman = genomes_pacman[i]
        genome_id_ghost_1, genome_ghost_1 = genomes_ghost_1[i]
        genome_id_ghost_2, genome_ghost_2 = genomes_ghost_2[i]

        netp = neat.nn.FeedForwardNetwork.create(genome_pacman, config)
        netg1 = neat.nn.FeedForwardNetwork.create(genome_ghost_1, config_ghost)
        netg2 = neat.nn.FeedForwardNetwork.create(genome_ghost_2, config_ghost)

        genome_pacman.fitness, genome_ghost_1.fitness, genome_ghost_2.fitness = play_game(netp, netg1, netg2, copy.deepcopy(PACMAN_BOARD), config)

    #
    # for genome_id, genome in genomes:
    #     net = neat.nn.FeedForwardNetwork.create(genome, config)
    #     genome.fitness = play_game(net, copy.deepcopy(PACMAN_BOARD), config)

def play_game(netPac, netGOne, netGTwo, map, config, print_end=False):
    """
    Plays a simulated game of pacman
    Args:
        net: The network created from the individual's genome,
             used to get Pacman's next direction
        map (int[][]): The 2D board array
    Returns:
        int: Score to be used for the fitness of the individual
    """
    score = [0,0,0]
    turns = 0
    pacPos = PAC_STARTING_POS
    ghostone = GHOSTONE
    ghosttwo = GHOSTTWO
    pacMappy = []
    pacVisited = set()
    ghostvisited = set()
    candies = 0
    distfromPacOne = 15
    distfromPacTwo = 15


    # Flattens a 2D array into a 1D
    # flat_map = sum(map, [])
    # Play for 50 turns or when Pacman gets all food

    while score[0] < config.fitness_threshold and turns < 100:

        if print_end:
            print(turns)
            for row in map:
                print(row)
            print()

        if pacPos == ghostone or pacPos == ghosttwo:
            return (-500 + turns + score[0], 500 - turns + score[1], 500 - turns + score[2])

        movedOntoGhost = False
        if pacPos not in pacVisited:
            score[0] += 2*UNVISITED_SCORE
            pacVisited.add(pacPos)
        if ghostone not in ghostvisited or ghosttwo not in ghostvisited:
            score[1] += UNVISITED_SCORE
            score[2] += UNVISITED_SCORE
            ghostvisited.add(ghosttwo)
            ghostvisited.add(ghostone)


        inputs = []
        nearestFood, fooddist = move_for_nearest_food(map,pacPos,pacVisited)
        inputs.append(fooddist)
        for move in dir_changes:
            if(move == nearestFood):
                inputs.append(1)
            else:
                inputs.append(0)

        nearGhostOne, ghostonedist = move_for_object(map, pacPos, ghostone)
        nearGhostOne = nearGhostOne[0]*-1, nearGhostOne[1]*-1
        inputs.append(ghostonedist)
        for move in dir_changes:
            if(move == nearGhostOne):
                inputs.append(1)
            else:
                inputs.append(0)

        nearGhostTwo, ghosttwodist = move_for_object(map, pacPos, ghosttwo)
        nearGhostTwo = nearGhostTwo[0]*-1,nearGhostTwo[1]*-1
        inputs.append(ghosttwodist)
        for move in dir_changes:
            if(move == nearGhostTwo):
                inputs.append(1)
            else:
                inputs.append(0)
        outputs = netPac.activate(inputs)


        dir = 0
        max_val = -100.0

        for i, output in enumerate(outputs):
            if output > max_val:
                max_val = output
                dir = i

        delta = dir_changes[dir]
        movePac = addCoord(pacPos, delta)

        if valid(map, movePac):
            if map[movePac[0]][movePac[1]] == GHOST:
                movedOntoGhost = True

        else:
            score[0] += WALL_SCORE

        # GHOST ONE NOW
        inputs = []
        movedOntoPacman = False

        inputs.append(ghostonedist)
        nearPacOne, dist = move_for_object(map, ghostone, pacPos)
        for move in dir_changes:
            if(move == nearPacOne):
                inputs.append(1)
            else:
                inputs.append(0)
        ghost_dists = move_for_object(map, ghostone, ghosttwo)[1]
        nearestFood, fooddist = move_for_nearest_food(map, ghostone, ghostvisited)
        for move in dir_changes:
            if(move == nearestFood):
                inputs.append(1)
            else:
                inputs.append(0)
        inputs.append(dist)

        inputs.append(ghost_dists)


        outputs = netGOne.activate(inputs)

        dir = 0
        max_val = -100.0

        for i, output in enumerate(outputs):
            if output > max_val:
                max_val = output
                dir = i

        delta = dir_changes[dir]
        move = addCoord(ghostone, nearPacOne)
        if valid(map, move):
            if map[move[0]][move[1]] == PACMAN:
                movedOntoPacman = True

            if ghostone not in pacVisited:
                map[ghostone[0]][ghostone[1]] = FOOD
            else:
                map[ghostone[0]][ghostone[1]] = EMPTY
            ghostone = move
            map[ghostone[0]][ghostone[1]] = GHOST

        else:
            score[1] += WALL_SCORE
        score[1] -= dist
        # GHOST TWO NOW
        inputs = []

        inputs.append(ghosttwodist)
        nearPacTwo, dist = move_for_object(map, ghosttwo, pacPos)
        for move in dir_changes:
            if(move == nearPacTwo):
                inputs.append(1)
            else:
                inputs.append(0)
        inputs.append(ghost_dists)
        nearestFood, fooddist = move_for_nearest_food(map, ghosttwo, ghostvisited)
        for move in dir_changes:
            if(move == nearestFood):
                inputs.append(1)
            else:
                inputs.append(0)
        inputs.append(dist)
        outputs = netGTwo.activate(inputs)

        dir = 0
        max_val = -100.0

        for i, output in enumerate(outputs):
            if output > max_val:
                max_val = output
                dir = i

        delta = dir_changes[dir]
        move = addCoord(ghosttwo, nearPacTwo)

        if valid(map, move):
            if map[move[0]][move[1]] == PACMAN:
                movedOntoPacman = True
            if ghosttwo not in pacVisited:
                map[ghosttwo[0]][ghosttwo[1]] = FOOD
            else:
                map[ghosttwo[0]][ghosttwo[1]] = EMPTY
            ghosttwo = move
            map[ghosttwo[0]][ghosttwo[1]] = GHOST
        else:
            score[2] += WALL_SCORE
        score[2] -= dist

        if (movedOntoGhost and movedOntoPacman):
            return (-500 + turns + score[0], 500 - turns+score[1], 500 - turns+score[2])

        if valid(map, movePac):
            if not movedOntoPacman:
                map[pacPos[0]][pacPos[1]] = EMPTY
            map[movePac[0]][movePac[1]] = PACMAN
            pacPos = movePac


        turns += 1

    return (score[0] + 500 - turns, -500 + score[1], score[2] -500)

def addCoord(tup1, tup2):
    (x,y) = tup1[0] + tup2[0], tup1[1] + tup2[1]
    return (x,y)

def move_for_object(map, current, target):
    """
    Returns the distance to the nearest food
    """
    ydist = abs(current[0] - target[0])
    xdist = abs(current [1] - target[1])
    dist = ydist + xdist
    if xdist >= ydist and xdist > 0:
        if current[1] >= target[1]:
            return (0,-1), dist
        else:
            return (0,1), dist
    #this is a weird case.. prolly cant actually happen
    elif xdist == ydist and xdist == 0:
        if current[0] >= target[0]:
            return (-1,0), dist
        else:
            return (1,0), dist
    else:
        if current[0] >= target[0]:
            return (-1,0), dist
        else:
            return (1,0), dist



def move_for_nearest_food(map, pacPos, v):
    """
    Returns the distance to the nearest food
    """
    q = [(pacPos,0)]
    pacVisited = set()
    path = {}
    path[pacPos] = None

    while len(q) > 0:
        currentPos,dist = q.pop(len(q)-1)
        pacVisited.add(currentPos)

        if currentPos not in v:
            while path[currentPos] != pacPos:
                currentPos = path[currentPos]
            direction = currentPos[0] - pacPos[0], currentPos[1] - pacPos[1]
            return direction, dist

        for new_pos in [(currentPos[0]+x, currentPos[1]+y) for x, y in dir_changes]:
            if new_pos not in pacVisited and valid(map, new_pos):
                q.insert(0, (new_pos, dist+1))
                path[new_pos] = currentPos


    return ((0,0),0)


def valid(map, pacPos):
    """
    Checks if a pacPosition is in-bounds of the array
    """
    if pacPos[0] >= 0 and pacPos[0] < len(PACMAN_BOARD) and pacPos[1] >= 0 and pacPos[1] < len(PACMAN_BOARD[0]):
        return True
    else:
        return False


def run(config_file, ghost_file):
    """
    Runs the NEAT experiment
    Args:
        config_file: The config file containing the parameters for the ANN
    """

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    config_ghost = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         ghost_file)

    p = TotalPopulation(config, config_ghost)

    p.add_reporter_pacman(neat.StdOutReporter(True))
    p.add_reporter_ghost_1(neat.StdOutReporter(True))
    p.add_reporter_ghost_2(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    stats2 = neat.StatisticsReporter()
    stats3 = neat.StatisticsReporter()
    p.add_reporter_pacman(stats)
    p.add_reporter_ghost_1(stats2)
    p.add_reporter_ghost_2(stats3)

    p.add_reporter_pacman(neat.Checkpointer(100))

    winnerp, winnerg1, winnerg2 = p.run(eval_genomes, 50)

    print('\nBest genome Pacman:\n{}'.format(winnerp))
    print('\nBest genome Ghost 1:\n{}'.format(winnerg1))
    print('\nBest genome Ghost 2:\n{}'.format(winnerg2))

    print('\nOutput:')
    net1 = neat.nn.FeedForwardNetwork.create(winnerp, config)
    net2 = neat.nn.FeedForwardNetwork.create(winnerg1, config_ghost)
    net3 = neat.nn.FeedForwardNetwork.create(winnerg2, config_ghost)
    # test the winner_net in some slowed down pacman game
    play_game(net1, net2, net3, PACMAN_BOARD, config, print_end=True)
    
    visualize.draw_net(config, winnerp, True, filename="Pacman_Net")
    visualize.draw_net(config_ghost, winnerg1, True, filename="Ghost_1_Net")
    visualize.draw_net(config_ghost, winnerg2, True, filename="Ghost_2_Net")
    visualize.plot_stats(stats, ylog=False, view=True, filename="Pacman_Stats.svg",title="Pacman's average and best fitness")
    visualize.plot_stats(stats2, ylog=False, view=True, filename="Ghost_1_Stats.svg", title="Ghost 1's average and best fitness")
    visualize.plot_stats(stats3, ylog=False, view=True, filename="Ghost_2_Stats.svg", title="Ghost 2's average and best fitness")
    visualize.plot_species(stats, view=True, title="Pacman Speciation", filename="Pacman_Speciation.svg")
    visualize.plot_species(stats2, view=True, title="Ghost 1 Speciation", filename="Ghost_1_Speciation.svg")
    visualize.plot_species(stats3, view=True, title="Ghost 2 Speciation", filename="Ghost_2_Speciation.svg")



if __name__ == '__main__':
    """
    The entry point of the program
    """
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'pacsim-params')
    ghost_path = os.path.join(local_dir, 'pacsim-ghost-params')
    run(config_path, ghost_path)

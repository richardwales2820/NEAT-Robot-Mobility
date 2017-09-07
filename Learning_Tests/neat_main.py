import visualize
import os
import neat
from copy import deepcopy

delta = [(0, 1), (0, -1), (1, 0), (-1, 0)]

WHEELS = 0
PADDLES = 1
MOB_REQUIRED = {'-': WHEELS, '*': PADDLES}
TURNS = 100
BOT = '@'

START_COURSE = [
    ['@','-','-','-','-','*','*','*','*','-'],
    ['-','-','-','-','-','*','*','*','-','-'],
    ['-','-','-','-','*','*','*','-','-','-'],
    ['-','-','-','-','*','*','*','-','-','-'],
    ['-','-','-','-','*','*','*','*','-','-'],
    ['-','-','-','-','*','*','*','*','*','-'],
    ['-','-','-','-','*','*','*','*','*','*'],
    ['-','-','-','-','-','-','-','-','-','-'],
    ['-','*','*','-','-','*','-','-','-','-'],
    ['-','*','*','*','-','*','-','-','-','-']
]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = run_course(net)

def run_course(net, final=False):
    state = WHEELS
    visited = set()
    fitness = 0
    pos = (0, 0)
    old_piece = '-'
    course = deepcopy(START_COURSE)
    
    for _ in range(TURNS):
        inputs = []

        if pos not in visited:
            fitness += 10
            visited.add(pos)

        # Look at adjacent points in the course
        for delt in delta:
            nx,ny = tuple(x+y for x,y in zip(pos, delt))
            
            if nx < 0 or ny < 0 or nx >= len(course) or ny >= len(course[0]):
                inputs.append(ord('#'))
                continue
            
            inputs.append(ord(course[nx][ny]))
            
        dir = 0
        max_val = -100.0
        outputs = net.activate(inputs)
        
        for i, output in enumerate(outputs[:4]):
            if output > max_val:
                max_val = output
                dir = i
        
        if outputs[4] < 0.5:
            state = WHEELS
        else:
            state = PADDLES
        
        nx,ny = (delta[dir][0] + pos[0], delta[dir][1] + pos[1])
        if nx < 0 or ny < 0 or nx >= len(course) or ny >= len(course[0]):
            continue
        
        try:
            if MOB_REQUIRED[course[nx][ny]] == state:
                course[pos[0]][pos[1]] = old_piece
                old_piece = course[nx][ny]
                
                pos = (nx, ny)
                course[nx][ny] = BOT
                
        except KeyError as e:
            print(course)
            raise e
    
    if final:
        for x,y in visited:
            course[x][y] = '$'
        for row in course:
            print(''.join(row) + '\n')
        print('Visited: {}'.format(visited))

    return fitness

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(300))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Use the winner_net to run through a course in slow-motion
    run_course(winner_net, final=True)

    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
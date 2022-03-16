from random import randrange
from numpy.random import randint, permutation


class GeneticTSP:
    def __init__(self, N=40, M=1000, gens=50):
        # Number of nodes in graph
        self.N = N
        # Population size
        self.M = M
        # Number of generations to be tried
        self.generations = gens

        self.best_fitness = None
        self.fittest = None

        self.edges = self.initialize_problem_space()
        self.population = self.initialize_population()
        self.adj_population = self.get_adjacency()

        self.simulate(self.generations)
        print(f"Best fitness: {self.best_fitness}, Fittest individual: {self.fittest}")
        print(self.edges)

    def simulate(self, generations=10):
        avg_fitness, best, best_ind = self.population_fitness()
        print(f"Average fitness of generation: {avg_fitness}, Best fitness: {best}")

        self.best_fitness = best
        self.fittest = best_ind

        for i in range(generations):
            self.population = self.populate_next_generation()  # O(n.m)
            self.adj_population = self.get_adjacency()  # O(n.log(n))

            avg_fitness, best, best_ind = self.population_fitness()  # O(n.m)
            if best < self.best_fitness:
                self.best_fitness = best
                self.fittest = best_ind
            print(f"Average fitness of generation: {avg_fitness}, Best fitness: {best}")

    # Initialize given number of nodes and random distance edges between them
    def initialize_problem_space(self):
        edges = []
        for i in range(self.N):
            dists = []
            for j in range(i + 1):
                if i == j:
                    dists.append(0)
                else:
                    dists.append(randrange(19) + 1)
            edges.append(dists)
        return edges

    # Get distance of the edge between two given nodes
    def get_edge(self, node1, node2):
        min_node = min(node1, node2)
        max_node = max(node1, node2)
        return self.edges[max_node][min_node]

    # Create sample population
    def initialize_population(self):
        population = []

        for i in range(self.M):
            population.append(list(permutation(self.N)))

        return population

    # Get adjacency representation for population
    def get_adjacency(self):
        adjacency_population = []
        for i in range(self.M):
            adjacency_population.append(self.get_adjacency_single(list(self.population[i])))
        return adjacency_population

    def get_adjacency_single(self, individual):
        current = individual.copy()
        for j in range(len(current)):
            current[j] = individual[(individual.index(j) + 1) % self.N]
        return current

    # Evaluate individual fitness
    def fitness(self, individual):
        distance = 0
        for start, end in enumerate(individual):
            distance += self.get_edge(start, end)
        return distance

    # Evaluate population fitness
    def population_fitness(self):
        total = 0
        max_fitness = "None"
        fittest_ind = []
        for ind in self.adj_population:
            cur_fitness = self.fitness(ind)
            total += cur_fitness
            if max_fitness == "None" or cur_fitness < max_fitness:
                max_fitness = cur_fitness
                fittest_ind = ind
        avg_fitness = total / self.M

        return avg_fitness, max_fitness, fittest_ind

    # Selection for breeding
    def selection(self, k=3):
        # first random selection
        selection_ix = randint(len(self.adj_population))
        for ix in randint(0, self.M, k - 1):
            # check if better (e.g. perform a tournament)
            if self.fitness(self.adj_population[ix]) < self.fitness(self.adj_population[selection_ix]):
                selection_ix = ix
        return self.adj_population[selection_ix]

    # Crossover 2 parents to get offspring using heuristic crossover
    def crossover(self, parent1, parent2):
        # Kid starts from random node
        start = randrange(self.N)
        visited = [start]
        # Repeat until visited full
        while len(visited) < self.N:
            # Compare parent edges leaving starting node and pick shorter
            dist1 = self.get_edge(visited[-1], parent1[visited[-1]])
            dist2 = self.get_edge(visited[-1], parent2[visited[-1]])
            if dist1 < dist2:
                # If shorter node in visited, select random unvisited edge
                if parent1[visited[-1]] in visited:
                    node_set = set(range(self.N))
                    visited_set = set(visited)
                    visited.append(list(node_set.difference(visited_set))[0])
                else:
                    visited.append(parent1[visited[-1]])
            else:
                # If shorter node in visited, select random unvisited edge
                if parent2[visited[-1]] in visited:
                    node_set = set(range(self.N))
                    visited_set = set(visited)
                    visited.append(list(node_set.difference(visited_set))[0])
                else:
                    visited.append(parent2[visited[-1]])
        return visited

    def populate_next_generation(self):
        new_population = []
        for i in range(self.M):
            parent1 = self.selection()
            parent2 = self.selection()
            offspring = self.crossover(parent1, parent2)
            new_population.append(offspring)
        return new_population

if __name__ == '__main__':
    GeneticTSP()

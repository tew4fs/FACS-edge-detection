from PIL import Image
import numpy as np
import random

from facs.image_processor import create_intensity_array
from config.constants import (
    NUMBER_OF_ANTS,
    NUMBER_OF_ITERATIONS,
    NUMBER_OF_CONSTRUCTIONS,
    PHEROMONE_DECAY_COEFFICIENT,
    INITIAL_PHEROMONE_VALUE,
    DEGREE_OF_EXPLORTATION,
    PARAMETER_INFLUENCING_HEURISTIC_INFORMATION,
    PARAMETER_INFLUENCING_PHEROMONE_TRAIL,
    PHEROMONE_EVAPORATION_COEFFICIENT,
)


def create_edge_detection_image(image):
    intensity_array = create_intensity_array(image)
    heuristics_array = get_heuristics_array(intensity_array)
    ranked_heuristics_map = get_ranked_heuristics_map(heuristics_array)
    global_pheromones = run_algorithm(intensity_array, ranked_heuristics_map)
    print(global_pheromones)
    final_image = Image.fromarray((global_pheromones).astype(np.uint8))
    return final_image


def run_algorithm(intensity_array, heuristics_map):
    initial_positions = list(heuristics_map.keys())[:NUMBER_OF_ANTS]
    image_height = intensity_array.shape[0]
    image_width = intensity_array.shape[1]
    ants = _initialize_ants(initial_positions)
    ant_visited_pixels = _initialize_ant_visited_pixels_dict()
    ant_local_pheromones = _initialize_ant_local_pheromones(image_height, image_width)
    global_pheromones = np.full((image_height, image_width), INITIAL_PHEROMONE_VALUE)
    for iteration in range(NUMBER_OF_ITERATIONS):
        print(f"Iteration: {iteration}")
        for construction in range(NUMBER_OF_CONSTRUCTIONS):
            print(f"Construction: {construction}")
            for ant in range(NUMBER_OF_ANTS):
                ant_location = ants[ant]
                visited_pixels = ant_visited_pixels[ant]
                pixel = move_pixel(
                    ant_location, image_height, image_width, visited_pixels, ant_local_pheromones[ant], heuristics_map
                )
                ant_visited_pixels[ant].append(ants[ant])
                ants[ant] = pixel
                ant_local_pheromones[ant] = _update_local_pheromones(
                    ant_local_pheromones[ant], ant_location[0], ant_location[1]
                )
        global_pheromones = _update_global_pheromones(global_pheromones, ant_local_pheromones, ant_visited_pixels)
    return global_pheromones


def move_pixel(ant_location, image_height, image_width, visited_pixels, local_pheromones, heuristics_map):
    row_index = ant_location[0]
    column_index = ant_location[1]
    neighbors = _get_neighbors(row_index, column_index, image_height, image_width)
    unvisited_neighbors = _get_unvisited_neighbors(neighbors, visited_pixels)
    uniform_random_value = random.uniform(0, 1)
    pixel = None
    if uniform_random_value <= DEGREE_OF_EXPLORTATION and unvisited_neighbors:
        pixel = _get_best_unvisited_neighbor(unvisited_neighbors, local_pheromones, heuristics_map)
    else:
        pixel = _explore_neighbors(neighbors)
    return pixel


def _update_local_pheromones(local_pheromone_array, row_index, column_index):
    updated_local_pheromone_value = (1 - PHEROMONE_DECAY_COEFFICIENT) * local_pheromone_array[
        row_index, column_index
    ] + PHEROMONE_DECAY_COEFFICIENT * INITIAL_PHEROMONE_VALUE
    local_pheromone_array[row_index][column_index] = updated_local_pheromone_value
    return local_pheromone_array


def _update_global_pheromones(global_pheromone_array, local_pheromones, ant_visited_pixels):
    image_height = global_pheromone_array.shape[0]
    image_width = global_pheromone_array.shape[1]
    for row_index in range(image_height):
        for column_index in range(image_width):
            average_pheromone_level = _compute_average_pheromone_level(
                row_index, column_index, local_pheromones, ant_visited_pixels
            )
            global_pheromone_array[row_index][column_index] = (1 - PHEROMONE_EVAPORATION_COEFFICIENT) + (
                PHEROMONE_EVAPORATION_COEFFICIENT * average_pheromone_level
            )
    return global_pheromone_array


def _compute_average_pheromone_level(row_index, column_index, ant_local_pheromones, ant_visited_pixels):
    pheromone_level = 0
    number_visited = 1
    for ant in range(NUMBER_OF_ANTS):
        if (row_index, column_index) in ant_visited_pixels[ant]:
            pheromone_level += ant_local_pheromones[ant][row_index][column_index]
            number_visited += 1
    return pheromone_level / number_visited


def _get_best_unvisited_neighbor(unvisited_neighbors, local_pheromones, heuristics_map):
    best_neighbor = unvisited_neighbors[0]
    best_neighbor_value = 0
    for neighbor in unvisited_neighbors:
        row_index = neighbor[0]
        column_index = neighbor[1]
        value = (
            local_pheromones[row_index][column_index] ** PARAMETER_INFLUENCING_PHEROMONE_TRAIL
            * heuristics_map[neighbor] ** PARAMETER_INFLUENCING_HEURISTIC_INFORMATION
        )
        if value > best_neighbor_value:
            best_neighbor_value = value
            best_neighbor = neighbor
    return best_neighbor


def _explore_neighbors(neighbors):
    index = random.randint(0, len(neighbors) - 1)
    return neighbors[index]


def get_ranked_heuristics_map(heuristics_array):
    pixel_to_heuristic_map = {}
    image_height = heuristics_array.shape[0]
    image_width = heuristics_array.shape[1]
    for row_index in range(image_height):
        for column_index in range(image_width):
            pixel_to_heuristic_map[(row_index, column_index)] = heuristics_array[row_index][column_index]
    ranked_pixel_to_heuristic_map = dict(sorted(pixel_to_heuristic_map.items(), key=lambda item: item[1], reverse=True))
    return ranked_pixel_to_heuristic_map


def get_heuristics_array(intensity_array):
    image_height = intensity_array.shape[0]
    image_width = intensity_array.shape[1]
    variation_array = _get_variation_array(intensity_array)
    max_variation = _get_max_variation(variation_array)
    heuristics_array = np.empty([image_height, image_width])
    for row_index in range(image_height):
        for column_index in range(image_width):
            heuristics_array[row_index][column_index] = variation_array[row_index][column_index] / max_variation
    return heuristics_array


def _get_variation_array(intensity_array):
    image_height = intensity_array.shape[0]
    image_width = intensity_array.shape[1]
    variation_array = np.empty([image_height, image_width], dtype=int)
    for row_index in range(image_height):
        for column_index in range(image_width):
            variation_array[row_index][column_index] = _get_variation_value(intensity_array, row_index, column_index)
    return variation_array


def _get_max_variation(variation_array):
    max_variation = 0
    for column in variation_array:
        for value in column:
            if value > max_variation:
                max_variation = value
    return max_variation


def _get_variation_value(intensity_array, row_index, column_index):
    variation_value = (
        abs(
            _get_intensity_value(intensity_array, row_index - 1, column_index - 1)
            - _get_intensity_value(intensity_array, row_index + 1, column_index + 1)
        )
        + abs(
            _get_intensity_value(intensity_array, row_index, column_index - 1)
            - _get_intensity_value(intensity_array, row_index, column_index + 1)
        )
        + abs(
            _get_intensity_value(intensity_array, row_index + 1, column_index - 1)
            - _get_intensity_value(intensity_array, row_index - 1, column_index + 1)
        )
        + abs(
            _get_intensity_value(intensity_array, row_index + 1, column_index)
            - _get_intensity_value(intensity_array, row_index - 1, column_index)
        )
    )
    return variation_value


def _get_intensity_value(intensity_array, row_index, column_index):
    image_height = intensity_array.shape[0]
    image_width = intensity_array.shape[1]
    if row_index < 0 or column_index < 0 or row_index >= image_height or column_index >= image_width:
        return 0
    intensity_value = intensity_array[row_index][column_index]
    return intensity_value


def _get_unvisited_neighbors(ant_visited_pixels, neighbors):
    unvisited_neighbors = []
    for neighbor in neighbors:
        if neighbor not in ant_visited_pixels:
            unvisited_neighbors.append(neighbor)
    return unvisited_neighbors


def _get_neighbors(row_index, column_index, max_height, max_width):
    neighbors = []
    if column_index - 1 >= 0:
        neighbors.append((row_index, column_index - 1))
    if column_index + 1 < max_width:
        neighbors.append((row_index, column_index + 1))
    if row_index - 1 >= 0:
        neighbors.append((row_index - 1, column_index))
        if column_index - 1 >= 0:
            neighbors.append((row_index - 1, column_index - 1))
        if column_index + 1 < max_width:
            neighbors.append((row_index - 1, column_index + 1))
    if row_index + 1 < max_height:
        neighbors.append((row_index + 1, column_index))
        if column_index - 1 >= 0:
            neighbors.append((row_index + 1, column_index - 1))
        if column_index + 1 < max_width:
            neighbors.append((row_index + 1, column_index + 1))
    return neighbors


def _initialize_ants(initial_positions):
    ants = {}
    for ant in range(NUMBER_OF_ANTS):
        ants[ant] = initial_positions[ant]
    return ants


def _initialize_ant_visited_pixels_dict():
    ant_visted_pixels = {}
    for ant in range(NUMBER_OF_ANTS):
        ant_visted_pixels[ant] = []
    return ant_visted_pixels


def _initialize_ant_local_pheromones(image_height, image_width):
    ant_local_pheromones = {}
    for ant in range(NUMBER_OF_ANTS):
        ant_local_pheromones[ant] = np.full((image_height, image_width), INITIAL_PHEROMONE_VALUE)
    return ant_local_pheromones

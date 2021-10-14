import logging
import math

from lux.constants import Constants
from lux.game import Game
from lux.game_map import Cell

DIRECTIONS = Constants.DIRECTIONS
game_state = None
logging.basicConfig(filename="agent.log", level=logging.INFO)
build_location = None
unit_to_city_dict = {}
unit_to_resource_dict = {}


def get_resource_tiles(game_state, height, width):
    resource_tiles: list[Cell] = []
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)
    return resource_tiles


def get_close_resource(unit, resource_tiles, player):
    closest_dist = math.inf
    closest_resource_tile = None
    # if the unit is a worker and we have space in cargo, lets find the nearest resource tile and try to mine it
    for resource_tile in resource_tiles:
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        if resource_tile in unit_to_resource_dict.values(): continue

        dist = resource_tile.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    return closest_resource_tile


def get_close_city(player, unit):
    closest_dist = math.inf
    closest_city_tile = None
    for k, city in player.cities.items():
        for city_tile in city.citytiles:
            dist = city_tile.pos.distance_to(unit.pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_city_tile = city_tile
    return closest_city_tile


def find_empty_tile_near(near_what, game_state, observation):
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for d in dirs:
        try:
            possible_empty_tile = game_state.map.get_cell(near_what.pos.x + d[0],
                                                          near_what.pos.y + d[1])
            logging.info(f"{observation['step']}: checking: {possible_empty_tile}")
            if possible_empty_tile.resource is None and possible_empty_tile.road == 0 and possible_empty_tile.citytile is None:
                build_location = possible_empty_tile
                logging.info(f"{observation['step']}: Found: {possible_empty_tile}")
                return build_location
        except Exception as e:
            logging.warning(f"{observation['step']}: while search: {str(e)}")
    dirs = [(1, -1), (-1, 1), (-1, -1), (1, 1)]
    for d in dirs:
        try:
            possible_empty_tile = game_state.map.get_cell(near_what.pos.x + d[0],
                                                          near_what.pos.y + d[1])
            logging.info(f"{observation['step']}: checking: {possible_empty_tile}")
            if possible_empty_tile.resource is None and possible_empty_tile.road == 0 and possible_empty_tile.citytile is None:
                build_location = possible_empty_tile
                logging.info(f"{observation['step']}: Found: {possible_empty_tile}")
                return build_location
        except Exception as e:
            logging.warning(f"{observation['step']}: while search: {str(e)}")
    return None


def agent(observation, configuration):
    global game_state
    global build_location
    global unit_to_resource_dict
    global unit_to_city_dict
    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    actions = []
    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height
    resource_tiles = get_resource_tiles(game_state, width, height)
    workers = [u for u in player.units if u.is_worker()]
    for w in workers:
        if w not in unit_to_city_dict:
            logging.info(f"Found worker unaccounted for {w}")
            city_assignment = get_close_city(player, w)
            unit_to_city_dict[w] = city_assignment
    for w in workers:
        if w not in unit_to_resource_dict:
            logging.info(f"Found worker w/o resource {w}")
            resource_assignment = get_close_resource(w, resource_tiles, player)
            unit_to_resource_dict[w] = resource_assignment
    logging.info(f" worker :{workers}")
    cities = player.cities.values()
    city_tiles = []
    for city in cities:
        for c_tile in city.citytiles:
            city_tiles.append(c_tile)
    logging.info(f"{cities}")
    logging.info(f"{city_tiles}")
    build_city = False
    if len(workers) / len(city_tiles) >= 0.75:
        build_city = True
    # we iterate over all our units and do something with them
    for unit in player.units:
        if unit.is_worker() and unit.can_act():
            if unit.get_cargo_space_left() > 0:
                intended_resource = unit_to_resource_dict[unit]
                cell = game_state.map.get_cell(intended_resource.pos.x, intended_resource.pos.y)
                if cell.has_resource():
                    actions.append(unit.move(unit.pos.direction_to(intended_resource.pos)))
                else:
                    intended_resource = get_close_resource(unit, resource_tiles, player)
                    unit_to_resource_dict[unit] = resource_assignment
                    actions.append(unit.move(unit.pos.direction_to(intended_resource.pos)))
            else:
                if build_city:
                    if build_location is None:
                        empty_near = get_close_city(player, unit)
                        build_location = find_empty_tile_near(empty_near, game_state, observation)
                    if unit.pos == build_location.pos:
                        action = unit.build_city()
                        actions.append(action)
                        build_city = False
                        build_location = None
                        continue
                    else:
                        logging.info(f"{observation['step']}: Navigating:")
                        actions.append(unit.move(unit.pos.direction_to(build_location.pos)))
                        continue
                # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
                elif len(player.cities) > 0:
                    if unit in unit_to_city_dict and unit_to_city_dict[unit] in city_tiles:
                        move_dir = unit.pos.direction_to(unit_to_city_dict[unit].pos)
                        actions.append(unit.move(move_dir))
                    else:
                        unit_to_city_dict[unit] = get_close_city(player, unit)
                        move_dir = unit.pos.direction_to(unit_to_city_dict[unit].pos)
                        actions.append(unit.move(move_dir))
    # you can add debug annotations using the functions in the annotate object
    # actions.append(annotate.circle(0, 0))
    return actions

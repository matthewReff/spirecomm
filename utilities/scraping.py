import logging
import os
import json

from spirecomm.spire.game import Game
from spirecomm.spire.relic import Relic
from spirecomm.spire.potion import Potion
from spirecomm.spire.card import Card
from spirecomm.spire.character import Monster, PlayerClass
from spirecomm.communication.action import *


def load_data_from_file(file_name):
    try:
        with open(file_name, "r") as json_file:
            jsonData = json.load(json_file)
            logging.debug(f"Loaded {len(jsonData)} items from {file_name}")

            return jsonData
    except Exception as e:
        logging.error("Unable to load", file_name, str(e))


class GameDataManager:
    file_name = ""
    data = dict()

    def __init__(self, file_name):
        self.file_name = file_name
        self.data = load_data_from_file(file_name)

    def save(self):
        try:
            with open(self.file_name, "w") as json_file:
                json.dump(self.data, json_file)
        except Exception as e:
            logging.error("Unable to save", self.file_name, str(e))

    def convert(self, key):
        try:
            return self.data[key]
        except Exception:
            logging.error("No entry found for", key)

    def attempt_update(self, key):
        try:
            if key not in self.data:
                self.data[key] = len(self.data) + 1
                self.save()
                logging.debug("Added new entry: " + key + " in " + str(self.data))
        except Exception as e:
            logging.error("Unable to update", str(e))


class Scraper:
    root_folder_path = ""
    card_data_manager = None
    relic_data_manager = None
    potion_data_manager = None
    monster_data_manager = None
    event_data_manager = None
    CARD_FILE_NAME = "cards.json"
    RELICS_FILE_NAME = "relics.json"
    POTIONS_FILE_NAME = "potions.json"
    MONSTERS_FILE_NAME = "monsters.json"
    EVENTS_FILE_NAME = "events.json"

    def __init__(self, class_to_load: PlayerClass, root_folder_path: str = None):
        if root_folder_path is None:
            current_directory = os.getcwd()
            self.root_folder_path = os.path.join(current_directory, "gameData")
        else:
            self.root_folder_path = root_folder_path
        self.__initialize_objects(self.root_folder_path, class_to_load)

    def __create_folder(self, folder_path: str):
        try:
            os.mkdir(folder_path)
        except FileExistsError:
            # Completely fine, that's what we wanted
            pass
        except Exception:
            logging.error("Unable to create folder: " + folder_path)

    def __create_file(self, file_path: str):
        try:
            os.mknod(file_path)
            # Fill with empty dict to make things happy
            with open(file_path, "w") as json_file:
                json.dump(dict(), json_file)
        except FileExistsError:
            # Completely fine, that's what we wanted
            pass
        except Exception:
            logging.error("Unable to create file: " + file_path)

    def __initialize_objects(self, root_folder_path: str, class_to_load: PlayerClass):
        self.__create_folder(root_folder_path)
        class_file_base = os.path.join(root_folder_path, class_to_load.name)
        self.__create_folder(class_file_base)

        card_path = os.path.join(class_file_base, self.CARD_FILE_NAME)
        self.__create_file(card_path)
        self.card_data_manager = GameDataManager(card_path)

        relic_path = os.path.join(class_file_base, self.RELICS_FILE_NAME)
        self.__create_file(relic_path)
        self.relic_data_manager = GameDataManager(relic_path)

        potion_path = os.path.join(class_file_base, self.POTIONS_FILE_NAME)
        self.__create_file(potion_path)
        self.potion_data_manager = GameDataManager(potion_path)

        monster_path = os.path.join(root_folder_path, self.MONSTERS_FILE_NAME)
        self.__create_file(monster_path)
        self.monster_data_manager = GameDataManager(monster_path)

        event_path = os.path.join(root_folder_path, self.EVENTS_FILE_NAME)
        self.__create_file(event_path)
        self.event_data_manager = GameDataManager(event_path)

    def change_class(self, class_to_load: PlayerClass):
        self.__initialize_objects(self.root_folder_path, class_to_load)

    def scrape_state(self, gameState: Game):
        try:
            self.__scrape_for_cards(gameState)
            self.__scrape_for_monsters(gameState)
            self.__scrape_for_potions(gameState)
            self.__scrape_for_relics(gameState)
            # TODO add events
        except Exception as e:
            logging.debug("Ran into error during scrape" + str(e))

    def __scrape_for_cards(self, gameState: Game):
        logging.debug("Scraping card data")
        try:
            for cardCollection in [
                gameState.draw_pile,
                gameState.discard_pile,
                gameState.exhaust_pile,
                gameState.hand,
            ]:
                card: Card
                for card in cardCollection:
                    self.card_data_manager.attempt_update(card.card_id)
        except Exception as e:
            logging.error("Ran into error while scraping for cards:" + str(e))

    def __scrape_for_monsters(self, gameState: Game):
        logging.debug("Scraping monster data")
        try:
            monster: Monster
            for monster in gameState.monsters:
                self.monster_data_manager.attempt_update(monster.monster_id)
        except Exception as e:
            logging.error("Ran into error while scraping for monsters:" + str(e))

    def __scrape_for_relics(self, gameState: Game):
        logging.debug("Scraping relic data")
        try:
            relic: Relic
            for relic in gameState.relics:
                self.relic_data_manager.attempt_update(relic.relic_id)
        except Exception as e:
            logging.error("Ran into error while scraping for relics:" + str(e))

    def __scrape_for_potions(self, gameState: Game):
        logging.debug("Scraping potion data")
        try:
            potion: Potion
            for potion in gameState.potions:
                self.potion_data_manager.attempt_update(potion.potion_id)
        except Exception as e:
            logging.error("Ran into error while scraping for potions:" + str(e))

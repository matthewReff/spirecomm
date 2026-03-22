import logging

from spirecomm.spire.game import Game
from spirecomm.spire.relic import Relic
from spirecomm.spire.potion import Potion
from spirecomm.spire.card import Card
from spirecomm.spire.character import Monster, PlayerClass
from spirecomm.communication.action import *
import pysqlite3


def get_class_name(player_class: PlayerClass) -> str:
    if player_class == PlayerClass.DEFECT:
        return "defect"
    elif player_class == PlayerClass.IRONCLAD:
        return "ironclad"
    elif player_class == PlayerClass.THE_SILENT:
        return "silent"
    else:
        raise Exception("Unknown class {}".format(player_class))


class EncodingDatabase:
    db_connection = None
    db_cursor = None
    player_class = None
    player_class_name = None

    def __init__(self, player_class: PlayerClass):
        self.player_class = player_class
        self.player_class_name = self.player_class_name
        self.db_connection = pysqlite3.connect("slay-ai")

    def upsert_tables(self):
        self.db_connection.execute(
            "CREATE TABLE IF NOT EXISTS card(id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR UNIQUE, player_class VARCHAR UNIQUE, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, CONSTRAINT player_scope UNIQUE (player_class, name) ON CONFLICT REPLACE)"
        )
        self.db_connection.execute(
            "CREATE TABLE IF NOT EXISTS artifact(id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR UNIQUE, player_class VARCHAR UNIQUE, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, CONSTRAINT player_scope UNIQUE (player_class, name) ON CONFLICT REPLACE)"
        )
        self.db_connection.execute(
            "CREATE TABLE IF NOT EXISTS potion(id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR UNIQUE, player_class VARCHAR UNIQUE, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, CONSTRAINT player_scope UNIQUE (player_class, name) ON CONFLICT REPLACE)"
        )

        self.db_connection.execute(
            "CREATE TABLE IF NOT EXISTS enemy(id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR UNIQUE, player_class VARCHAR UNIQUE, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )

    def save_card(self, name: str):
        self.db_connection.execute(
            'INSERT OR IGNORE INTO card(name, player_class) VALUES("{}", "{}")'.format(
                name, self.player_class_name
            )
        )

    def get_card(self, name: str):
        result = self.db_connection.execute(
            'SELECT * FROM card where name="{}" AND player_class="{}"'.format(
                name, self.player_class_name
            )
        )
        return result.fetchone()

    def save_artifact(self, name: str):
        self.db_connection.execute(
            'INSERT OR IGNORE INTO artifact(name, player_class) VALUES("{}", "{}")'.format(
                name, self.player_class_name
            )
        )

    def get_artifact(self, name: str):
        result = self.db_connection.execute(
            'SELECT * FROM artifact where name="{}" AND player_class="{}"'.format(
                name, self.player_class_name
            )
        )
        return result.fetchone()

    def save_potion(self, name: str):
        self.db_connection.execute(
            'INSERT OR IGNORE INTO potion(name, player_class) VALUES("{}", "{}")'.format(
                name, self.player_class_name
            )
        )

    def get_potion(self, name: str):
        result = self.db_connection.execute(
            'SELECT * FROM potion where name="{}" AND player_class="{}"'.format(
                name, self.player_class_name
            )
        )
        return result.fetchone()

    def save_enemy(self, name: str):
        self.db_connection.execute(
            'INSERT OR IGNORE INTO enemy(name) VALUES("{}")'.format(name)
        )

    def get_enemy(self, name: str):
        result = self.db_connection.execute(
            'SELECT * FROM enemy where name="{}"'.format(name)
        )
        return result.fetchone()


class EncodingMapper:
    encoding_database = None

    def __init__(self, encoding_database: EncodingDatabase):
        self.encoding_database = encoding_database

    # Take in a new game state, add any newly seen items into encoding
    def scrape_state(self, gameState: Game):
        self.__scrape_for_cards(gameState)
        self.__scrape_for_monsters(gameState)
        self.__scrape_for_potions(gameState)
        self.__scrape_for_relics(gameState)

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
                    self.encoding_database.save_card(card.name)
        except Exception as e:
            logging.error("Ran into error while scraping for cards:" + str(e))

    def __scrape_for_monsters(self, gameState: Game):
        logging.debug("Scraping monster data")
        try:
            monster: Monster
            for monster in gameState.monsters:
                self.encoding_database.save_enemy(monster.name)
        except Exception as e:
            logging.error("Ran into error while scraping for monsters:" + str(e))

    def __scrape_for_relics(self, gameState: Game):
        logging.debug("Scraping relic data")
        try:
            relic: Relic
            for relic in gameState.relics:
                self.encoding_database.save_artifact(relic.name)
        except Exception as e:
            logging.error("Ran into error while scraping for relics:" + str(e))

    def __scrape_for_potions(self, gameState: Game):
        logging.debug("Scraping potion data")
        try:
            potion: Potion
            for potion in gameState.potions:
                self.encoding_database.save_potion(potion.name)
        except Exception as e:
            logging.error("Ran into error while scraping for potions:" + str(e))

    def get_card_encoding(self, name) -> int:
        card = self.encoding_database.get_card(name)
        return card.id

    def get_artifact_encoding(self, name) -> int:
        artifact = self.encoding_database.get_artifact(name)
        return artifact.id

    def get_potion_encoding(self, name) -> int:
        potion = self.encoding_database.get_potion(name)
        return potion.id

    def get_enemy_encoding(self, name) -> int:
        enemy = self.encoding_database.get_enemy(name)
        return enemy.id

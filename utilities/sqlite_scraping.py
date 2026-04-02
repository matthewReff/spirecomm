import logging

from spirecomm.spire.power import Power
from spirecomm.spire.game import Game
from spirecomm.spire.relic import Relic
from spirecomm.spire.potion import Potion
from spirecomm.spire.character import Monster, Player, PlayerClass
import sqlite3


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
        self.player_class_name = get_class_name(self.player_class)
        self.db_connection = sqlite3.connect("slay-ai.db")

    def _upsert_tables(self):
        self.db_connection.execute(
            "CREATE TABLE IF NOT EXISTS card(id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, player_class VARCHAR, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, CONSTRAINT player_scope UNIQUE (player_class, name) ON CONFLICT IGNORE)"
        )
        self.db_connection.execute(
            "CREATE TABLE IF NOT EXISTS relic(id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, player_class VARCHAR, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, CONSTRAINT player_scope UNIQUE (player_class, name) ON CONFLICT IGNORE)"
        )
        self.db_connection.execute(
            "CREATE TABLE IF NOT EXISTS potion(id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, player_class VARCHAR, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, CONSTRAINT player_scope UNIQUE (player_class, name) ON CONFLICT IGNORE)"
        )
        self.db_connection.execute(
            "CREATE TABLE IF NOT EXISTS power(id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, player_class VARCHAR, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, CONSTRAINT player_scope UNIQUE (player_class, name) ON CONFLICT IGNORE)"
        )

        self.db_connection.execute(
            "CREATE TABLE IF NOT EXISTS monster(id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR UNIQUE ON CONFLICT IGNORE, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self.db_connection.commit()

    def _drop_all(self):
        self.db_connection.execute("drop table card")
        self.db_connection.execute("drop table relic")
        self.db_connection.execute("drop table potion")
        self.db_connection.execute("drop table power")
        self.db_connection.execute("drop table monster")
        self.db_connection.commit()

    def save_card(self, name: str):
        logging.debug("Saving mapping for card " + name)
        self.db_connection.execute(
            'INSERT INTO card(name, player_class) VALUES("{}", "{}") ON CONFLICT(name, player_class) DO NOTHING'.format(
                name, self.player_class_name
            )
        )
        self.db_connection.commit()

    def get_card(self, name: str):
        logging.debug("Grabbing mapping for card " + name)
        result = self.db_connection.execute(
            'SELECT * FROM card where name="{}" AND player_class="{}"'.format(
                name, self.player_class_name
            )
        )
        id, card_name, player_class, created_at = result.fetchone()
        return id

    def _debug_cards(self):
        result = self.db_connection.execute("SELECT * FROM card")
        return result.fetchall()

    def save_relic(self, name: str):
        logging.debug("Saving mapping for relic " + name)
        self.db_connection.execute(
            'INSERT INTO relic(name, player_class) VALUES("{}", "{}") ON CONFLICT(name, player_class) DO NOTHING'.format(
                name, self.player_class_name
            )
        )
        self.db_connection.commit()

    def get_relic(self, name: str):
        logging.debug("Grabbing mapping for relic " + name)
        result = self.db_connection.execute(
            'SELECT * FROM relic where name="{}" AND player_class="{}"'.format(
                name, self.player_class_name
            )
        )
        id, relic_name, player_class, created_at = result.fetchone()
        return id

    def _debug_relics(self):
        result = self.db_connection.execute("SELECT * FROM relic")
        return result.fetchall()

    def save_potion(self, name: str):
        logging.debug("Saving mapping for potion " + name)
        self.db_connection.execute(
            'INSERT INTO potion(name, player_class) VALUES("{}", "{}") ON CONFLICT(name, player_class) DO NOTHING'.format(
                name, self.player_class_name
            )
        )
        self.db_connection.commit()

    def get_potion(self, name: str):
        logging.debug("Grabbing mapping for potion " + name)
        result = self.db_connection.execute(
            'SELECT * FROM potion where name="{}" AND player_class="{}"'.format(
                name, self.player_class_name
            )
        )
        id, potion_name, player_class, created_at = result.fetchone()
        return id

    def _debug_potions(self):
        result = self.db_connection.execute("SELECT * FROM potion")
        return result.fetchall()

    def save_power(self, name: str):
        logging.debug("Saving mapping for power " + name)
        self.db_connection.execute(
            'INSERT INTO power(name, player_class) VALUES("{}", "{}") ON CONFLICT(name, player_class) DO NOTHING'.format(
                name, self.player_class_name
            )
        )
        self.db_connection.commit()

    def get_power(self, name: str):
        logging.debug("Grabbing mapping for power " + name)
        result = self.db_connection.execute(
            'SELECT * FROM power where name="{}" AND player_class="{}"'.format(
                name, self.player_class_name
            )
        )
        id, power_name, player_class, created_at = result.fetchone()
        return id

    def _debug_powers(self):
        result = self.db_connection.execute("SELECT * FROM power")
        return result.fetchall()

    def save_monster(self, name: str):
        logging.debug("Saving mapping for monster " + name)
        self.db_connection.execute(
            'INSERT INTO monster(name) VALUES("{}")'.format(name)
        )
        self.db_connection.commit()

    def get_monster(self, name: str):
        logging.debug("Grabbing mapping for monster " + name)
        result = self.db_connection.execute(
            'SELECT * FROM monster where name="{}"'.format(name)
        )
        id, monster_name, created_at = result.fetchone()
        return id

    def _debug_monsters(self):
        result = self.db_connection.execute("SELECT * FROM monster")
        return result.fetchall()


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
        self.__scrape_for_powers(gameState)

    def __scrape_for_cards(self, gameState: Game):
        logging.debug("Scraping card data")
        all_cards = []
        all_cards = all_cards + gameState.draw_pile
        all_cards = all_cards + gameState.discard_pile
        all_cards = all_cards + gameState.exhaust_pile
        all_cards = all_cards + gameState.hand

        try:
            for card in all_cards:
                logging.debug("Attempting to add {} to db".format(card.name))
                self.encoding_database.save_card(card.name)
        except Exception as e:
            logging.error("Ran into error while scraping for cards:" + str(e))

    def __scrape_for_monsters(self, gameState: Game):
        logging.debug("Scraping monster data")
        try:
            monster: Monster
            for monster in gameState.monsters:
                self.encoding_database.save_monster(monster.name)
        except Exception as e:
            logging.error("Ran into error while scraping for monsters:" + str(e))

    def __scrape_for_relics(self, gameState: Game):
        logging.debug("Scraping relic data")
        try:
            relic: Relic
            for relic in gameState.relics:
                self.encoding_database.save_relic(relic.name)
        except Exception as e:
            logging.error("Ran into error while scraping for relics:" + str(e))

    def __scrape_for_powers(self, gameState: Game):
        logging.debug("Scraping power data")

        playerData: Player | None = gameState.player
        monsters: list[Monster] = gameState.monsters

        all_powers = []
        if playerData is not None:
            logging.critical("player has " + str(len(playerData.powers)) + " power(s)")
            all_powers = all_powers + [power for power in playerData.powers]
        for monster in monsters:
            logging.critical(
                monster.name + " has " + str(len(monster.powers)) + " power(s)"
            )
            monster_powers = [power for power in monster.powers]
            all_powers = all_powers + monster_powers

        for power in all_powers:
            logging.critical(power.power_name + power.power_id)

        try:
            power: Power
            for power in all_powers:
                self.encoding_database.save_power(power.power_name)
        except Exception as e:
            logging.error("Ran into error while scraping for powers:" + str(e))

    def __scrape_for_potions(self, gameState: Game):
        logging.debug("Scraping potion data")
        try:
            potion: Potion
            for potion in gameState.potions:
                self.encoding_database.save_potion(potion.name)
        except Exception as e:
            logging.error("Ran into error while scraping for potions:" + str(e))

    def get_card_encoding(self, name) -> int:
        card_id = self.encoding_database.get_card(name)
        return int(card_id)

    def get_relic_encoding(self, name) -> int:
        relic_id = self.encoding_database.get_relic(name)
        return int(relic_id)

    def get_potion_encoding(self, name) -> int:
        potion_id = self.encoding_database.get_potion(name)
        return int(potion_id)

    def get_power_encoding(self, name) -> int:
        power_id = self.encoding_database.get_power(name)
        return int(power_id)

    def get_monster_encoding(self, name) -> int:
        monster_id = self.encoding_database.get_monster(name)
        return int(monster_id)

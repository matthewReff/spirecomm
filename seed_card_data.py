import os

from spirecomm.spire.character import PlayerClass
from utilities.scraping import GameDataManager
from spirecomm.ai.priorities import (
    DefectPowerPriority,
    SilentPriority,
    IroncladPriority,
)


# Run this after initial configuration has been ran so the folders + files exist
def main():
    current_directory = os.getcwd()
    root_folder_path = os.path.join(current_directory, "../../SlayTheSpire/gameData")

    for class_name in [play_class for play_class in PlayerClass]:
        class_file_base_path = os.path.join(root_folder_path, class_name.name)

        CARD_FILE_NAME = "cards.json"
        RELICS_FILE_NAME = "relics.json"

        card_path = os.path.join(class_file_base_path, CARD_FILE_NAME)
        card_data_manager = GameDataManager(card_path)

        relic_path = os.path.join(class_file_base_path, RELICS_FILE_NAME)
        relic_data_manager = GameDataManager(relic_path)

        priority_set = get_priority_for_class(class_name)

        for card_name in priority_set.CARD_PRIORITY_LIST:
            card_data_manager.attempt_update(card_name)

        for relic_name in priority_set.BOSS_RELIC_PRIORITIES:
            relic_data_manager.attempt_update(relic_name)


T


def get_priority_for_class(player_class: PlayerClass):
    if player_class == PlayerClass.THE_SILENT:
        return SilentPriority()
    elif player_class == PlayerClass.IRONCLAD:
        return IroncladPriority()
    elif player_class == PlayerClass.DEFECT:
        return DefectPowerPriority()


if __name__ == "__main__":
    main()

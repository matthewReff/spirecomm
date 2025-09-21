import random
import time
import os
import logging

from spirecomm.communication.coordinator import Coordinator
from spirecomm.ai.nnAgent import NnAgent
from spirecomm.ai.agent import Agent
from spirecomm.spire.character import PlayerClass


def main():
    logging.basicConfig(filename="neuralNet.log", level=logging.DEBUG)
    agent: Agent = NnAgent()
    coordinator = Coordinator()
    coordinator.signal_ready()
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)

    # We're running an AI, it doesn't make sense to play anything other than defect
    chosenClass = PlayerClass.DEFECT
    agent.change_class(chosenClass)
    while True:
        seed = str(int(time.time()))
        save_name = "SlayAI"
        random.seed(seed)

        logging.info("Starting run playing " + str(chosenClass) + " with seed " + seed)
        results = coordinator.climb_till_defeat(chosenClass, seed)

        folder_name = save_name + "-" + str(chosenClass) + "-" + seed
        copy_run_files(results, chosenClass, folder_name)


def get_class_folder_name(chosenClass: PlayerClass) -> str:
    if chosenClass == PlayerClass.IRONCLAD:
        return "1_IRONCLAD"
    elif chosenClass == PlayerClass.THE_SILENT:
        return "1_THE_SILENT"
    elif chosenClass == PlayerClass.DEFECT:
        return "1_DEFECT"


def copy_run_files(results, chosenClass, folder_name):
    # Quick and dirty grab of run files for a given ascension streak
    # We assume that the CWD is the SlayTheSpire game folder
    repo_base = os.path.join(os.getcwd(), "..")
    game_path = os.path.join(repo_base, "SlayTheSpire")
    mod_path = os.path.join(repo_base, "Mods", "spirecomm")

    # We are located in the SlayTheSpire directory by default
    game_runs_path = os.path.join(game_path, "runs", get_class_folder_name(chosenClass))

    mod_runs_path = os.path.join(mod_path, "runs")
    mod_specific_runs_path = os.path.join(mod_runs_path, folder_name)

    logging.info("Creating runs folder in mod folder")
    try:
        os.mkdir(mod_runs_path)
        logging.info("Created path: " + mod_runs_path, "continuing...")
    except FileExistsError:
        logging.info("Path already exists: " + mod_runs_path, "continuing...")

    logging.info("Creating specific run folder in mod runs folder")
    try:
        os.mkdir(mod_specific_runs_path)
        logging.info("Created path: " + mod_specific_runs_path, "continuing...")
    except FileExistsError:
        logging.info("Path already exists: " + mod_specific_runs_path, "continuing...")

    try:
        logging.info("Copying from game runs folder to mod runs folder")
        run_files = os.listdir(game_runs_path)

        run_files.sort()
        run_file_names = run_files[-len(results) :]

        for file in run_file_names:
            with open(os.path.join(game_runs_path, file), "r") as source_data:
                with open(os.path.join(mod_specific_runs_path, file), "w") as dest_data:
                    dest_data.write(source_data.read())
    except OSError as e:
        logging.error(
            "Ran into a problem while copying run information from game to mod folder: "
            + e
        )


if __name__ == "__main__":
    main()

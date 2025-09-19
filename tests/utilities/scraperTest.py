import unittest

from spirecomm.spire.character import PlayerClass

from utilities.scraping import Scraper

class TestScraper(unittest.TestCase):
	def test_creates_folders_on_start(self):
		scraper = Scraper(PlayerClass.DEFECT)

	def loads_existing_data(self):
		scraper = Scraper(PlayerClass.DEFECT)
		cardManager = scraper.card_data_manager
		self.assertIsNotNone(cardManager)
		self.assertGreater(len(cardManager.data), 0)

if __name__ == '__main__':
	unittest.main()
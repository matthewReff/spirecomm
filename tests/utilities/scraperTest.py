import unittest

from utilities.sqlite_scraping import EncodingDatabase, EncodingMapper
from spirecomm.spire.character import PlayerClass

class TestScraper(unittest.TestCase):
    def test_database_upsert(self):
        db = EncodingDatabase(PlayerClass.DEFECT)
        db.upsert_tables()

    def loads_existing_data(self):
        db = EncodingDatabase(PlayerClass.DEFECT)
        encoding_mapper = EncodingMapper(db)

        a = encoding_mapper.get_card_encoding("a")
        print(a)

if __name__ == "__main__":
    unittest.main()

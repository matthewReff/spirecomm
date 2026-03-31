import unittest

from utilities.sqlite_scraping import EncodingDatabase, EncodingMapper
from spirecomm.spire.character import PlayerClass


class TestEncoder(unittest.TestCase):
    def test_database_upsert(self):
        db = EncodingDatabase(PlayerClass.DEFECT)
        db._upsert_tables()

    def loads_existing_data(self):
        db = EncodingDatabase(PlayerClass.DEFECT)
        encoding_mapper = EncodingMapper(db)

        example_card_encoding = encoding_mapper.get_card_encoding("example_card")
        print(example_card_encoding)


if __name__ == "__main__":
    unittest.main()

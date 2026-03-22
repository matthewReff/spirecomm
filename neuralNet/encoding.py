from sklearn.preprocessing import LabelEncoder

potion_names = []
potion_encoder = LabelEncoder()
potion_encoder.fit(potion_names)

card_encoder = LabelEncoder()
card_names = []
card_encoder.fit(card_names)

relic_encoder = LabelEncoder()
relic_names = []
relic_encoder.fit(relic_names)

enemy_encoder = LabelEncoder()
enemy_names = []
enemy_encoder.fit(enemy_names)

## Figure out how to auto-encode things as we discover new cards
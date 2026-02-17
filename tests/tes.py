# %%
# loading this tests/test_output tests/test_output/apa_test/train_positive_pairs.csv and tests/test_output/apa_test/test_positive_pairs.csv
import pandas as pd
train_positive_pairs = pd.read_csv('tests/test_output/apa_test/train_positive_pairs.csv')
test_positive_pairs = pd.read_csv('tests/test_output/apa_test/test_positive_pairs.csv')
print(train_positive_pairs.head())
print(test_positive_pairs.head())
# %%
# check "Animal Limb" is in the train_positive_pairs and "Animal Limb" is in the test_positive_pairs term1 column and if yes print full row
train_animal_limb = train_positive_pairs[train_positive_pairs['term1'] == 'Blood Platelets']
test_animal_limb = test_positive_pairs[test_positive_pairs['term1'] == 'Blood Platelets']
print(train_animal_limb)
# %%

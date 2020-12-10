import pandas as pd
from sklearn.model_selection import train_test_split

headers = ['sentence_id', 'text', 'position']
filename = '../Data Folder/sentences_just_one_position.csv'
dataframe = pd.read_csv(filename, sep='\t', names=headers)
train_percentage = 0.7
validation_percentage = 0.2
test_percentage = 0.1

val_plus_test = validation_percentage + test_percentage
train, val_and_test = train_test_split(
                dataframe, test_size=val_plus_test, random_state=7)
val_test_percentage = test_percentage/val_plus_test
val, test = train_test_split(val_and_test,
                             test_size=val_test_percentage, random_state=7)
train.to_csv("train.csv", sep='\t')
val.to_csv("val.csv", sep='\t')
test.to_csv("test.csv", sep='\t')

print("train")
print(train['position'].value_counts(normalize=True))
print("test")
print(test['position'].value_counts(normalize=True))
print("val")
print(val['position'].value_counts(normalize=True))
print("dataframe")
print(dataframe['position'].value_counts(normalize=True))

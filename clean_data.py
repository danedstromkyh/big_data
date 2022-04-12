import pandas as pd


def main():
    male_df = pd.read_csv('male.csv')
    female_df = pd.read_csv('female.csv')

    cleaned_male = male_df[['stature', 'weightkg', 'chestcircumference', 'waistcircumference', 'buttockcircumference', 'crotchheight']]
    cleaned_female = female_df[['stature', 'weightkg', 'chestcircumference', 'waistcircumference', 'buttockcircumference', 'crotchheight']]

    cleaned_male.to_csv('cleaned_male.csv', index=False)
    cleaned_female.to_csv('cleaned_female.csv', index=False)


if __name__ == '__main__':
    main()

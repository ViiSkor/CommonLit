import pandas as pd
from sklearn.model_selection import train_test_split

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('omw-1.4')

from models.train import choose_model
from models.preprocessing import preprocess


def get_data():
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")
    return train_df, test_df


if __name__ == '__main__':
    train_df, test_df = get_data()
    train_df['prep_x'] = preprocess(train_df)
    X = train_df['prep_x']
    y = train_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    training_results = choose_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)




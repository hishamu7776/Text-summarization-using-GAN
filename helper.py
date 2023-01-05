import os
import glob
import pandas as pd


def read_articles(articles_path, summaries_path, category_list=None, encoding='ISO-8859-1'):
    '''
      This function takes paths as argument and return a dataframe of text and summaries.
      This function does not work if there are no categories in specified folders.
      Required to write other functions if the folder structures are different.
    '''

    articles = list()
    summaries = list()
    if category_list is None:
        category_list = os.listdir(articles_path)

    for category in category_list:
        article_paths = glob.glob(os.path.join(
            articles_path, category, '*.txt'), recursive=True)
        summary_paths = glob.glob(os.path.join(
            summaries_path, category, '*.txt'), recursive=True)

        print(
            f'found {len(article_paths)} file in articles/{category} folder, {len(summary_paths)} file in summaries/{category}')

        if (len(article_paths) != len(summary_paths)):
            print("Summaries and articles are not matching.")
            return
        for idx, article_path in enumerate(article_paths):
            with open(article_path, mode='r', encoding=encoding) as file:
                articles.append(file.read())

            with open(summary_paths[idx], mode='r', encoding=encoding) as file:
                summaries.append(file.read())
    print(f'total {len(articles)} articles with {len(summaries)} summaries under {len(category_list)} specified category.')
    text_dataframe = pd.DataFrame(
        {'articles': articles, 'summaries': summaries},)
    return text_dataframe


def clean_dataframe(df):
    df.iloc[:, 0] = df.iloc[:, 0].str.encode(
        'ascii', 'ignore').str.decode('ascii')
    df.iloc[:, 1] = df.iloc[:, 1].str.encode(
        'ascii', 'ignore').str.decode('ascii')
    df = df.dropna()

import pandas as pd

class Movie_Dataset:
    def __init__(self, dataset_path):
        self.dataset=pd.read_csv(dataset_path,
                                sep="::",
                                encoding="latin1",
                                names=['movie_id', 'title', 'genres'],
                                header=None,
                                engine='python')
        
        

class User_Dataset:
    def __init__(self, dataset_path):
        self.dataset=pd.read_csv(
            dataset_path,
            sep="::",
            header=None,
            names=['user_id','gender','age','occupation','zip'],
            engine='python'
        )

class Rating_Dataset:
    def __init__(self, dataset_path):
        self.dataset=pd.read_csv(
            dataset_path,
            sep="::",
            names=['user_id','movie_id','rating','timestamp'],
            header=None,
            engine='python'
        )

        
        
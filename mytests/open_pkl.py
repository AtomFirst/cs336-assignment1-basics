import pickle

def get(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)
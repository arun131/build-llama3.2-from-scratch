import sentencepiece as spm
import os


class Tokenizer:
    def __init__(self, model_name: str):
        "Initialize the tokenizer"
        self.model_name = model_name
        self.repo_id = model_name
        self.filename = "tokenizer.model"
        self.director = model_name.split("/")[-1]

    def load_tokenizer_file(file_name:str):
        

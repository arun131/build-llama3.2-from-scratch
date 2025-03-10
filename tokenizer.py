import sentencepiece as spm
from dotenv import load_dotenv
import os
from huggingface_hub import hf_hub_download, login

load_dotenv()
login(token=os.getenv("HF_ACCESS_TOKEN"))
print("Hugging Face login successful")

class Tokenizer:
    def __init__(self, cls, model_name: str, filename: str) -> None:
        "Initialize the tokenizer"
        self.model_name = model_name
        self.repo_id = model_name
        self.filename = "tokenizer.model"
        self.local_dir = model_name.split("/")[-1]
        cls.load_tokenizer_file(self.filename)

    @classmethod
    def load_tokenizer_file(cls, model_name: str, filename: str) -> str:
        "Load the tokenizer file"
        tokenizer_file_name = hf_hub_download(
            repo_id=model_name,
            filename="tokenizer.model",
            )
        return tokenizer_file_name

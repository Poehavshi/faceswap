from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    models_dir: str = "/data/models/models"


settings = Settings()

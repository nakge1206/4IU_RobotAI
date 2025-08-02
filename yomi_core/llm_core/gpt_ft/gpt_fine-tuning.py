from openai import OpenAI
from pathlib import Path
import openai
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
settings = Settings()

def fine_tune(train_file_id, validation_file_id):
    try:
        # 파인 튜닝 작업을 생성하고 결과를 저장합니다.
        fine_tune_job = client.fine_tuning.jobs.create(
            model="gpt-4o-2024-08-06",
            training_file=train_file_id,
            validation_file = validation_file_id
        )
        
        # 생성된 파인 튜닝 작업의 ID를 사용하여 상태를 검색합니다.
        job_id = fine_tune_job.id  # 작업 ID를 얻습니다.
        print(f"Fine-tune job ID: {job_id}")  # 작업 ID를 출력합니다.

        # 작업 ID를 사용하여 파인 튜닝 작업의 상태를 검색합니다.
        job_status = client.fine_tuning.jobs.retrieve(job_id)
        print(job_status)  # 작업 상태를 출력합니다.
        
    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)
        
    
client = OpenAI(
    api_key = settings.openai_api_key,
)

train_data_file = client.files.create(
    file=Path("robot_core/fine_tuning_data/infp.jsonl"),
    purpose="fine-tune",
)
validation_data_file = client.files.create(
    file=Path("robot_core/fine_tuning_data/infp_va.jsonl"),
    purpose="fine-tune",
)

print(f"File ID: {train_data_file.id}")
print(f"File ID: {validation_data_file.id}")

fine_tune(train_data_file.id, validation_data_file.id)
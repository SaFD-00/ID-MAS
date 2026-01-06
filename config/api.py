"""
API configuration and credentials.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# .env 파일 로드 (기존 환경 변수를 덮어쓰기)
load_dotenv(PROJECT_ROOT / ".env", override=True)

# API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

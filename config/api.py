"""API 자격증명 및 환경 변수 설정 모듈.

이 모듈은 프로젝트에서 사용하는 외부 API 자격증명을 관리합니다.
.env 파일에서 환경 변수를 로드하여 API 키를 안전하게 관리합니다.

Attributes:
    PROJECT_ROOT: 프로젝트 루트 디렉토리 경로
    OPENAI_API_KEY: OpenAI API 키 (환경 변수에서 로드)
    HF_TOKEN: HuggingFace 토큰 (환경 변수에서 로드)

사용 예시:
    >>> from config.api import OPENAI_API_KEY, HF_TOKEN
    >>> if OPENAI_API_KEY:
    ...     print("OpenAI API 키가 설정되어 있습니다")
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리 (config/ 상위 폴더)
PROJECT_ROOT = Path(__file__).parent.parent

# .env 파일 로드 (기존 환경 변수를 덮어씀)
load_dotenv(PROJECT_ROOT / ".env", override=True)

# API 키 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

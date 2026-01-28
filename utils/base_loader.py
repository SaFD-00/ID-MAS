"""기본 데이터셋 로더 모듈 - 데이터셋 로더의 추상 인터페이스.

이 모듈은 모든 데이터셋 로더가 구현해야 하는 추상 인터페이스를 정의합니다.
다양한 답변 타입(MCQ, 숫자, LaTeX, 텍스트, Boolean)을 지원합니다.

주요 클래스:
    AnswerType: 답변 타입 열거형
    QuestionData: 통합 질문 데이터 구조
    BaseDatasetLoader: 추상 데이터셋 로더 기본 클래스

사용 예시:
    >>> from utils.base_loader import AnswerType, QuestionData
    >>> question = QuestionData(
    ...     dataset="gsm8k",
    ...     question_id="001",
    ...     question="What is 2+2?",
    ...     answer_type=AnswerType.NUMERIC,
    ...     ground_truth=4,
    ...     ground_truth_formatted="4"
    ... )
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional


class AnswerType(Enum):
    """답변 타입 열거형.

    다양한 데이터셋에서 지원하는 답변 타입을 정의합니다.

    Attributes:
        MCQ: 객관식 (A/B/C/D)
        NUMERIC: 숫자형 - 정수 또는 소수 (GSM8K, SVAMP, SciBench)
        LATEX: LaTeX 수식 (MATH)
        TEXT: 자유 형식 텍스트 (BBH)
        BOOLEAN: Yes/No, True/False (BBH)
    """
    MCQ = "mcq"
    NUMERIC = "numeric"
    LATEX = "latex"
    TEXT = "text"
    BOOLEAN = "boolean"


@dataclass
class QuestionData:
    """통합 질문 데이터 구조.

    모든 데이터셋에 대해 일관된 질문 인터페이스를 제공하는 데이터클래스입니다.
    소스 데이터셋에 관계없이 동일한 구조로 질문 데이터를 관리합니다.

    Attributes:
        dataset: 데이터셋 이름 (예: "gsm8k", "mmlu")
        question_id: 데이터셋 내 고유 식별자
        question: 질문 텍스트
        answer_type: 예상 답변 타입
        ground_truth: 원본 정답 값
        ground_truth_formatted: Teacher 모델 평가용 사람이 읽을 수 있는 형식
        choices: MCQ 타입에만 해당하는 선택지 리스트
        metadata: 데이터셋별 메타데이터
    """
    dataset: str
    question_id: str
    question: str
    answer_type: AnswerType
    ground_truth: Any
    ground_truth_formatted: str
    choices: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """직렬화를 위해 딕셔너리로 변환합니다.

        Returns:
            모든 필드를 포함하는 딕셔너리
        """
        return {
            "dataset": self.dataset,
            "question_id": self.question_id,
            "question": self.question,
            "answer_type": self.answer_type.value,
            "ground_truth": self.ground_truth,
            "ground_truth_formatted": self.ground_truth_formatted,
            "choices": self.choices,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionData":
        """딕셔너리에서 QuestionData 인스턴스를 생성합니다.

        Args:
            data: QuestionData 필드를 포함하는 딕셔너리

        Returns:
            생성된 QuestionData 인스턴스
        """
        return cls(
            dataset=data["dataset"],
            question_id=data["question_id"],
            question=data["question"],
            answer_type=AnswerType(data["answer_type"]),
            ground_truth=data["ground_truth"],
            ground_truth_formatted=data["ground_truth_formatted"],
            choices=data.get("choices"),
            metadata=data.get("metadata", {})
        )


class BaseDatasetLoader(ABC):
    """데이터셋 로더의 추상 기본 클래스.

    모든 데이터셋 로더가 구현해야 하는 인터페이스를 정의합니다.

    구현해야 하는 속성:
        dataset_name: HuggingFace 데이터셋 식별자
        answer_type: 데이터셋의 주요 답변 타입
        domain_category: 'math_logic' 또는 'science_knowledge'

    구현해야 하는 메서드:
        load_data(): HuggingFace에서 데이터 로드
        format_question_as_prompt(): 질문을 LLM 프롬프트로 변환
        format_ground_truth(): Teacher 평가용 정답 포맷팅
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """HuggingFace 데이터셋 식별자 (예: 'openai/gsm8k')."""
        pass

    @property
    @abstractmethod
    def answer_type(self) -> AnswerType:
        """이 데이터셋의 주요 답변 타입."""
        pass

    @property
    @abstractmethod
    def domain_category(self) -> str:
        """도메인 카테고리: 'math_logic' 또는 'science_knowledge'."""
        pass

    @abstractmethod
    def load_data(
        self,
        split: str = "train",
        subset: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[QuestionData]:
        """HuggingFace에서 데이터셋을 로드합니다.

        Args:
            split: 데이터 분할 ('train', 'validation', 'test')
            subset: 데이터셋 서브셋 (예: MMLU 과목, BBH 서브태스크)
            limit: 로드할 최대 질문 수

        Returns:
            QuestionData 객체 리스트
        """
        pass

    @abstractmethod
    def format_question_as_prompt(self, question: QuestionData) -> str:
        """질문을 LLM 입력용 프롬프트로 포맷팅합니다.

        Args:
            question: QuestionData 객체

        Returns:
            포맷팅된 프롬프트 문자열
        """
        pass

    @abstractmethod
    def format_ground_truth(self, question: QuestionData) -> str:
        """Teacher 모델 평가용 정답을 포맷팅합니다.

        Args:
            question: QuestionData 객체

        Returns:
            사람이 읽을 수 있는 정답 문자열
        """
        pass

    def get_learning_objective(self, subset: Optional[str] = None) -> str:
        """이 데이터셋의 학습 목표를 생성합니다.

        Args:
            subset: 더 구체적인 목표를 위한 선택적 서브셋 이름

        Returns:
            학습 목표 문자열
        """
        return f"Solve {self.dataset_name} problems with high accuracy"

    def get_available_subsets(self) -> Optional[List[str]]:
        """이 데이터셋에서 사용 가능한 서브셋 목록을 반환합니다.

        Returns:
            서브셋 이름 리스트, 해당 없으면 None
        """
        return None

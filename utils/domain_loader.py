"""도메인 기반 데이터셋 로더 모듈 - ID-MAS용.

이 모듈은 로컬 JSON 파일에서 도메인 데이터를 로드하기 위한 통합 인터페이스를 제공합니다.
각 학습 데이터셋은 별도 학습을 위한 고유한 Instructional Goal을 가집니다.

주요 클래스:
    DomainLoader: 도메인 기반 데이터 로더

지원 도메인:
    - math: GSM8K, MATH (학습) + SVAMP, ASDiv, MAWPS (평가)
    - logical: ReClor (학습) + ANLI R2/R3, BBH (평가)
    - commonsense: ARC-C (학습) + StrategyQA, OpenBookQA (평가)

사용 예시:
    >>> from utils.domain_loader import DomainLoader
    >>> loader = DomainLoader("math")
    >>> train_data = loader.load_training_data(dataset="gsm8k", limit=100)
    >>> eval_data = loader.load_eval_data("svamp", limit=50)
    >>> instructional_goal = loader.get_learning_objective("gsm8k")
"""
import json
import random
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from utils.base_loader import BaseDatasetLoader, QuestionData, AnswerType
from utils.answer_extractor import extract_boxed_answer


# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class DomainLoader(BaseDatasetLoader):
    """로컬 JSON 파일용 도메인 기반 데이터 로더.

    현재 지원하는 도메인:
        - math: GSM8K, MATH (학습) + SVAMP, ASDiv, MAWPS (평가)
        - logical: ReClor (학습) + ANLI R2/R3, BBH (평가)
        - commonsense: ARC-C (학습) + StrategyQA, OpenBookQA (평가)

    새 도메인은 DOMAIN_CONFIG를 확장하여 추가할 수 있습니다.
    각 학습 데이터셋은 고유한 Instructional Goal을 가지며 별도로 학습됩니다.
    평가 데이터는 데이터셋당 단일 파일에서 로드됩니다.
    """

    # Instructional Goals for each training dataset
    INSTRUCTIONAL_GOALS = {
        # Math domain
        "gsm8k": "Generate coherent, step-by-step mathematical reasoning in natural language that leads to a correct numerical answer for grade-school level math problems.",
        "math": "Solve advanced mathematical problems by selecting appropriate mathematical concepts and constructing logically valid, multi-step reasoning that leads to a correct solution.",

        # Logical domain
        "reclor": "Analyze logical reasoning problems by comprehending complex passages, identifying logical relationships, and selecting the most appropriate conclusion based on formal reasoning principles.",

        # Commonsense domain
        "arc_c": "Apply commonsense scientific knowledge to solve elementary science problems by understanding fundamental concepts and selecting the correct answer from multiple choices.",

        # BBH (Big Bench Hard)
        "bbh": "Evaluate and solve various logical reasoning tasks including boolean expressions, formal fallacies, logical deduction, and object tracking.",
    }

    DOMAIN_CONFIG = {
        "math": {
            "training_datasets": {
                "gsm8k": {"filename": "gsm8k_train.json", "answer_type": AnswerType.NUMERIC},
                "math": {"filename": "math_train.json", "answer_type": AnswerType.LATEX},
            },
            "eval_datasets": {
                "gsm8k": {"filename": "gsm8k_test.json", "answer_type": AnswerType.NUMERIC},
                "math": {"filename": "math_test.json", "answer_type": AnswerType.LATEX},
                "svamp": {"filename": "svamp_test.json", "answer_type": AnswerType.NUMERIC},
                "asdiv": {"filename": "asdiv_test.json", "answer_type": AnswerType.NUMERIC},
                "mawps": {"filename": "mawps_test.json", "answer_type": AnswerType.LATEX},  # 분수 포함
            },
            "default_answer_type": AnswerType.NUMERIC,
            "domain_category": "math_logic",
            "data_dir": "data/math"
        },
        "logical": {
            "training_datasets": {
                "reclor": {"filename": "reclor_train.json", "answer_type": AnswerType.MCQ},
            },
            "eval_datasets": {
                "reclor": {"filename": "reclor_test.json", "answer_type": AnswerType.MCQ},
                "anli_r2": {"filename": "anli_r2_test.json", "answer_type": AnswerType.MCQ},
                "anli_r3": {"filename": "anli_r3_test.json", "answer_type": AnswerType.MCQ},
                # BBH unified (all subtasks combined, subtask info in metadata.subtask)
                "bbh": {"filename": "bbh_test.json", "answer_type": AnswerType.TEXT},
            },
            "default_answer_type": AnswerType.MCQ,
            "domain_category": "logical_reasoning",
            "data_dir": "data/logical"
        },
        "commonsense": {
            "training_datasets": {
                "arc_c": {"filename": "arc_c_train.json", "answer_type": AnswerType.MCQ},
            },
            "eval_datasets": {
                "arc_c": {"filename": "arc_c_test.json", "answer_type": AnswerType.MCQ},
                "strategyqa": {"filename": "strategyqa_test.json", "answer_type": AnswerType.BOOLEAN},
                "openbookqa": {"filename": "openbookqa_test.json", "answer_type": AnswerType.MCQ},
            },
            "default_answer_type": AnswerType.MCQ,
            "domain_category": "commonsense_reasoning",
            "data_dir": "data/commonsense"
        }
    }

    def __init__(self, domain: str):
        """도메인 로더를 초기화합니다.

        Args:
            domain: 도메인 이름 (예: "math", "logical", "commonsense")

        Raises:
            ValueError: 알 수 없는 도메인인 경우
        """
        domain = domain.lower()
        if domain not in self.DOMAIN_CONFIG:
            raise ValueError(
                f"Unknown domain: {domain}. "
                f"Available domains: {list(self.DOMAIN_CONFIG.keys())}"
            )

        self.domain = domain
        self.config = self.DOMAIN_CONFIG[domain]
        self.data_dir = PROJECT_ROOT / self.config["data_dir"]
        self._current_eval_dataset: Optional[str] = None

    @property
    def dataset_name(self) -> str:
        """도메인 이름을 데이터셋 식별자로 반환합니다."""
        return f"domain_{self.domain}"

    @property
    def answer_type(self) -> AnswerType:
        """이 도메인의 기본 답변 타입을 반환합니다."""
        return self.config["default_answer_type"]

    @property
    def domain_category(self) -> str:
        """도메인 카테고리를 반환합니다."""
        return self.config["domain_category"]

    def load_data(
        self,
        split: str = "train",
        subset: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[QuestionData]:
        """분할에 따라 데이터를 로드합니다.

        Args:
            split: 학습 데이터는 "train", 평가는 "test"
            subset: train 분할의 경우 학습 데이터셋 지정, test의 경우 평가 데이터셋 지정
            limit: 최대 질문 수

        Returns:
            QuestionData 객체 리스트
        """
        if split == "train":
            if not subset:
                raise ValueError("Training dataset must be specified via 'subset' parameter")
            return self.load_training_data(dataset=subset, limit=limit)
        else:
            eval_dataset = subset or self._get_default_eval()
            return self.load_eval_data(eval_dataset, limit=limit)

    def load_training_data(
        self,
        dataset: str,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> List[QuestionData]:
        """이 도메인의 특정 학습 데이터셋을 로드합니다.

        각 학습 데이터셋(GSM8K, MATH, ReClor, ARC-C)은 고유한 Instructional Goal을
        가지므로 별도로 로드됩니다.

        Args:
            dataset: 학습 데이터셋 이름 (예: "gsm8k", "math", "reclor", "arc_c")
            limit: 로드할 최대 질문 수
            shuffle: 데이터 셔플 여부 (기본값: False)

        Returns:
            QuestionData 객체 리스트

        Raises:
            ValueError: 알 수 없는 학습 데이터셋인 경우
            FileNotFoundError: 학습 파일이 존재하지 않는 경우
        """
        dataset = dataset.lower()
        training_datasets = self.config["training_datasets"]

        if dataset not in training_datasets:
            raise ValueError(
                f"Unknown training dataset '{dataset}' for {self.domain} domain. "
                f"Available: {list(training_datasets.keys())}"
            )

        ds_config = training_datasets[dataset]
        filename = self._get_filename(ds_config)
        # New structure: data/{domain}/train/data/{dataset}_train.json
        file_path = self.data_dir / "train" / "data" / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Training file not found: {file_path}")

        questions = self._load_json_file(file_path, dataset, "train")
        print(f"  Loaded {len(questions)} questions from {filename}")

        # Shuffle data
        if shuffle and questions:
            random.shuffle(questions)
            print(f"  Shuffled {len(questions)} training questions")

        # Apply limit after shuffling
        if limit and len(questions) > limit:
            questions = questions[:limit]

        print(f"[{self.domain.upper()}/{dataset.upper()}] Loaded {len(questions)} training questions")
        return questions

    def load_enhanced_training_data(
        self,
        dataset: str,
        enhanced_data_dir: Path,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> List[QuestionData]:
        """Instructional Goal과 Task Analysis가 포함된 향상된 학습 데이터를 로드합니다.

        향상된 데이터 파일의 명명 패턴:
            {dataset}_train_ID-MAS.json

        이 파일들은 다음을 포함하는 instruction 필드를 가집니다:
            - 원본 instruction
            - Instructional Goal
            - Task Analysis (하위 기술 및 하위 과제)

        Args:
            dataset: 학습 데이터셋 이름 (예: "gsm8k", "math", "reclor", "arc_c")
            enhanced_data_dir: Enhanced data 디렉토리 경로
                (예: outputs/{domain}/train/{student_short}/data/)
            limit: 로드할 최대 질문 수
            shuffle: 데이터 셔플 여부 (기본값: False)

        Returns:
            QuestionData 객체 리스트

        Raises:
            FileNotFoundError: 향상된 데이터 파일이 존재하지 않는 경우
            ValueError: 이 도메인의 유효한 학습 데이터셋이 아닌 경우

        Example:
            >>> loader = DomainLoader("math")
            >>> data = loader.load_enhanced_training_data(
            ...     dataset="gsm8k",
            ...     enhanced_data_dir=Path("outputs/math/train/Qwen3-4B/data"),
            ...     limit=100
            ... )
        """
        dataset = dataset.lower()
        training_datasets = self.config["training_datasets"]

        if dataset not in training_datasets:
            raise ValueError(
                f"Unknown training dataset '{dataset}' for {self.domain} domain. "
                f"Available: {list(training_datasets.keys())}"
            )

        # Enhanced data filename pattern
        filename = f"{dataset}_train_ID-MAS.json"
        file_path = enhanced_data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Enhanced training file not found: {file_path}\n"
                f"Run design phase first to generate enhanced data."
            )

        questions = self._load_json_file(file_path, dataset, "train")
        print(f"  Loaded {len(questions)} enhanced questions from {filename}")

        # Shuffle data
        if shuffle and questions:
            random.shuffle(questions)
            print(f"  Shuffled {len(questions)} enhanced training questions")

        # Apply limit after shuffling
        if limit and len(questions) > limit:
            questions = questions[:limit]

        print(f"[{self.domain.upper()}/{dataset.upper()}] Loaded {len(questions)} enhanced training questions")
        return questions

    def get_available_enhanced_data(self, dataset: str = None) -> List[Dict[str, str]]:
        """이 도메인에서 사용 가능한 향상된 데이터 파일 목록을 반환합니다.

        outputs/{domain}/train/{student_short}/data/ 디렉토리에서 탐색합니다.

        Args:
            dataset: 데이터셋 이름으로 필터링 (선택사항)

        Returns:
            'dataset', 'student_suffix', 'path' 키를 가진 딕셔너리 리스트
        """
        from config.domains import OUTPUT_DIR

        train_dir = OUTPUT_DIR / self.domain / "train"
        if not train_dir.exists():
            return []

        results = []
        for model_dir in train_dir.iterdir():
            if not model_dir.is_dir():
                continue
            data_dir = model_dir / "data"
            if not data_dir.exists():
                continue
            for file_path in data_dir.glob("*_train_ID-MAS.json"):
                ds_name = file_path.stem.replace("_train_ID-MAS", "")
                if dataset is None or ds_name.lower() == dataset.lower():
                    results.append({
                        "dataset": ds_name,
                        "student_suffix": model_dir.name,
                        "path": str(file_path)
                    })

        return results

    def load_eval_data(
        self,
        dataset: str,
        limit: Optional[int] = None
    ) -> List[QuestionData]:
        """특정 평가 데이터셋을 로드합니다.

        Args:
            dataset: 평가 데이터셋 이름 (예: "gsm8k", "svamp", "arc_c")
            limit: 최대 질문 수

        Returns:
            QuestionData 객체 리스트

        Raises:
            ValueError: 알 수 없는 평가 데이터셋인 경우
            FileNotFoundError: 평가 파일이 존재하지 않는 경우
        """
        dataset = dataset.lower()
        if dataset not in self.config["eval_datasets"]:
            raise ValueError(
                f"Unknown eval dataset '{dataset}' for {self.domain} domain. "
                f"Available: {list(self.config['eval_datasets'].keys())}"
            )

        self._current_eval_dataset = dataset
        ds_config = self.config["eval_datasets"][dataset]
        filename = self._get_filename(ds_config)
        # New structure: data/{domain}/eval/data/{dataset}_test.json
        file_path = self.data_dir / "eval" / "data" / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Evaluation file not found: {file_path}")

        questions = self._load_json_file(file_path, dataset, "test")

        if limit and len(questions) > limit:
            questions = questions[:limit]

        print(f"[{self.domain.upper()}] Loaded {len(questions)} eval questions from {dataset}")
        return questions

    def _get_filename(self, ds_config) -> str:
        """데이터셋 설정에서 파일명을 추출합니다.

        Args:
            ds_config: 문자열(파일명) 또는 'filename' 키를 가진 딕셔너리

        Returns:
            파일명 문자열
        """
        if isinstance(ds_config, str):
            return ds_config
        return ds_config.get("filename", "")

    def _get_dataset_answer_type(self, dataset_name: str) -> AnswerType:
        """특정 데이터셋의 답변 타입을 반환합니다.

        Args:
            dataset_name: 데이터셋 이름

        Returns:
            해당 데이터셋의 AnswerType
        """
        dataset_name = dataset_name.lower()

        # Check training_datasets
        if dataset_name in self.config.get("training_datasets", {}):
            ds_config = self.config["training_datasets"][dataset_name]
            if isinstance(ds_config, dict) and "answer_type" in ds_config:
                return ds_config["answer_type"]

        # Check eval_datasets
        if dataset_name in self.config.get("eval_datasets", {}):
            ds_config = self.config["eval_datasets"][dataset_name]
            if isinstance(ds_config, dict) and "answer_type" in ds_config:
                return ds_config["answer_type"]

        # Fallback: use default_answer_type
        return self.config.get("default_answer_type", AnswerType.TEXT)

    def _load_json_file(
        self,
        file_path: Path,
        dataset_name: str,
        split: str
    ) -> List[QuestionData]:
        """JSON 파일을 로드하여 QuestionData 객체로 파싱합니다.

        Args:
            file_path: JSON 파일 경로
            dataset_name: 데이터셋 이름
            split: 데이터 분할 (train/test)

        Returns:
            QuestionData 객체 리스트
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            items = json.load(f)

        questions = []
        for i, item in enumerate(items):
            question_data = self._parse_item(
                item=item,
                dataset_name=dataset_name,
                question_id=f"{dataset_name}_{split}_{i}",
            )
            questions.append(question_data)

        return questions

    def _parse_item(
        self,
        item: Dict[str, Any],
        dataset_name: str,
        question_id: str
    ) -> QuestionData:
        """JSON 아이템을 QuestionData로 파싱합니다.

        예상 JSON 형식:
            {
                "instruction": "시스템 프롬프트",
                "input": "질문 텍스트",
                "output": "추론...\n\n#### 답변"
            }

        Args:
            item: JSON 아이템 딕셔너리
            dataset_name: 데이터셋 이름
            question_id: 고유 질문 식별자

        Returns:
            QuestionData 객체
        """
        question_text = item.get("input", "")
        output = item.get("output", "")
        instruction = item.get("instruction", "")

        # Extract answer from output using #### pattern
        answer, answer_type = self._extract_answer(output, dataset_name)

        # Detect MCQ choices from question text
        choices = self._extract_choices(question_text)

        return QuestionData(
            dataset=dataset_name,
            question_id=question_id,
            question=question_text,
            answer_type=answer_type,
            ground_truth=answer,
            ground_truth_formatted=f"The correct answer is {answer}",
            choices=choices,
            metadata={
                "instruction": instruction,
                "full_output": output
            }
        )

    def _extract_answer(self, output: str, dataset_name: str) -> tuple:
        """output 필드에서 답변을 추출하고 타입을 결정합니다.

        우선순위:
            1. \\boxed{...} 패턴 (중첩 중괄호 처리)
            2. #### 패턴 (GSM8K 스타일)
            3. 폴백: 마지막 줄

        Args:
            output: 답변이 포함된 출력 텍스트
            dataset_name: 데이터셋 이름 (타입 조회용)

        Returns:
            (answer, AnswerType) 튜플
        """
        # 1. Try \boxed{...} pattern first (handles nested braces)
        boxed_answer = extract_boxed_answer(output)
        if boxed_answer:
            answer = boxed_answer
        # 2. Try #### pattern (GSM8K style)
        elif match := re.search(r'####\s*(.+?)$', output.strip(), re.MULTILINE):
            answer = match.group(1).strip()
        # 3. Fallback: use last line
        else:
            answer = output.strip().split('\n')[-1].strip()

        # Get answer type from dataset config (not inferred)
        answer_type = self._get_dataset_answer_type(dataset_name)

        return answer, answer_type

    def _extract_choices(self, question_text: str) -> Optional[List[str]]:
        """질문 텍스트에서 객관식 선택지를 추출합니다.

        Args:
            question_text: 선택지가 포함될 수 있는 질문 텍스트

        Returns:
            선택지 리스트, 객관식이 아니면 None
        """
        # Pattern: A. choice text or A) choice text
        pattern = r'^([A-E])[.)]\s*(.+)$'
        lines = question_text.strip().split('\n')

        choices = []
        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                choices.append(match.group(2).strip())

        return choices if len(choices) >= 2 else None

    def _get_default_eval(self) -> str:
        """도메인의 기본 평가 데이터셋을 반환합니다."""
        eval_datasets = list(self.config["eval_datasets"].keys())
        return eval_datasets[0] if eval_datasets else None

    def get_available_eval_datasets(self) -> List[str]:
        """이 도메인에서 사용 가능한 평가 데이터셋 목록을 반환합니다."""
        return list(self.config["eval_datasets"].keys())

    def get_available_training_datasets(self) -> List[str]:
        """이 도메인의 학습 데이터셋 목록을 반환합니다."""
        return list(self.config["training_datasets"].keys())

    def format_ground_truth(self, question: QuestionData) -> str:
        """Teacher 모델 평가용 정답을 포맷팅합니다.

        Args:
            question: QuestionData 객체

        Returns:
            사람이 읽을 수 있는 정답 문자열
        """
        return question.ground_truth_formatted

    def get_learning_objective(self, dataset: str) -> str:
        """특정 학습 데이터셋의 Instructional Goal(학습 목표)을 반환합니다.

        각 학습 데이터셋은 고유한 Instructional Goal을 가집니다:
            - GSM8K: 초등 수학 단계별 추론
            - MATH: 고급 수학 문제 해결
            - ReClor: 논리적 추론 및 분석
            - ARC-C: 상식 과학 문제 해결

        Args:
            dataset: 학습 데이터셋 이름 (gsm8k, math, reclor, arc_c)

        Returns:
            해당 데이터셋의 Instructional Goal 문자열

        Raises:
            ValueError: 알 수 없는 데이터셋인 경우
        """
        dataset = dataset.lower()
        if dataset not in self.INSTRUCTIONAL_GOALS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available: {list(self.INSTRUCTIONAL_GOALS.keys())}"
            )
        return self.INSTRUCTIONAL_GOALS[dataset]

    def get_available_subsets(self) -> Optional[List[str]]:
        """사용 가능한 평가 데이터셋을 서브셋으로 반환합니다."""
        return self.get_available_eval_datasets()


# 외부 사용을 위한 편의 함수
def get_domain_loader(domain: str) -> DomainLoader:
    """도메인 로더 인스턴스를 반환합니다."""
    return DomainLoader(domain)


def get_available_domains() -> List[str]:
    """사용 가능한 도메인 목록을 반환합니다."""
    return list(DomainLoader.DOMAIN_CONFIG.keys())


def get_eval_datasets_for_domain(domain: str) -> List[str]:
    """도메인의 평가 데이터셋 목록을 반환합니다."""
    return DomainLoader(domain).get_available_eval_datasets()


if __name__ == "__main__":
    # Test the loader
    print("=" * 60)
    print("Testing DomainLoader")
    print("=" * 60)

    # Test math domain
    print("\n[Math Domain]")
    math_loader = DomainLoader("math")
    print(f"  Available eval datasets: {math_loader.get_available_eval_datasets()}")
    print(f"  Training datasets: {math_loader.get_available_training_datasets()}")

    # Test Instructional Goals
    for dataset in math_loader.get_available_training_datasets():
        print(f"\n  Instructional Goal for {dataset.upper()}:")
        print(f"    {math_loader.get_learning_objective(dataset)[:80]}...")

    # Load a few training samples from GSM8K
    print("\n  Loading GSM8K training data (limit=3):")
    try:
        train_data = math_loader.load_training_data(dataset="gsm8k", limit=3)
        print(f"  Sample training questions ({len(train_data)}):")
        for q in train_data:
            print(f"    [{q.dataset}] {q.question[:50]}... -> {q.ground_truth}")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")

    # Test evaluation data loading
    print("\n[Evaluation Data Test]")
    print("  Testing cross-dataset evaluation with SVAMP:")
    try:
        svamp_data = loader.load_eval_data("svamp", limit=3)
        print(f"  Loaded {len(svamp_data)} SVAMP questions")
        for q in svamp_data:
            print(f"    [{q.dataset}] {q.question[:50]}... -> {q.ground_truth}")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")

"""LLMWrapper 하위 호환성 모듈.

이 모듈은 하위 호환성을 위해 유지됩니다.
LLMWrapper와 GPTWrapper는 TeacherModelWrapper로 통합되었습니다.

Deprecated:
    새 코드에서는 TeacherModelWrapper를 직접 사용하세요.

변경 전 (deprecated):
    >>> from models.llm_wrapper import LLMWrapper
    >>> wrapper = LLMWrapper(config)

변경 후 (권장):
    >>> from models.teacher_wrapper import TeacherModelWrapper
    >>> wrapper = TeacherModelWrapper(config)

Warning:
    LLMWrapper 또는 GPTWrapper 사용 시 DeprecationWarning이 발생합니다.
"""
import warnings
from models.teacher_wrapper import TeacherModelWrapper, _fix_json_escapes


class LLMWrapper(TeacherModelWrapper):
    """TeacherModelWrapper의 deprecated 별칭.

    하위 호환성을 위해 유지되는 클래스입니다.
    인스턴스화 시 DeprecationWarning이 발생합니다.

    Deprecated:
        TeacherModelWrapper를 직접 사용하세요.
    """

    def __init__(self, *args, **kwargs):
        """LLMWrapper를 초기화합니다.

        DeprecationWarning을 출력하고 TeacherModelWrapper를 초기화합니다.

        Args:
            *args: TeacherModelWrapper로 전달되는 위치 인자
            **kwargs: TeacherModelWrapper로 전달되는 키워드 인자
        """
        warnings.warn(
            "LLMWrapper is deprecated, use TeacherModelWrapper instead",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


# 하위 호환성 별칭
GPTWrapper = LLMWrapper

__all__ = ["LLMWrapper", "GPTWrapper", "_fix_json_escapes"]

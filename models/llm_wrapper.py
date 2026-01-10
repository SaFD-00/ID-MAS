"""
Deprecated: LLMWrapper는 TeacherModelWrapper로 통합됨

이 모듈은 하위 호환성을 위해 유지됩니다.
새 코드에서는 TeacherModelWrapper를 직접 사용하세요.

Example:
    # Old (deprecated)
    from models.llm_wrapper import LLMWrapper
    wrapper = LLMWrapper(config)

    # New (recommended)
    from models.teacher_wrapper import TeacherModelWrapper
    wrapper = TeacherModelWrapper(config)
"""
import warnings
from models.teacher_wrapper import TeacherModelWrapper, _fix_json_escapes


class LLMWrapper(TeacherModelWrapper):
    """
    Deprecated: TeacherModelWrapper를 사용하세요.

    이 클래스는 하위 호환성을 위해 유지됩니다.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LLMWrapper is deprecated, use TeacherModelWrapper instead",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


# Backward compatibility alias
GPTWrapper = LLMWrapper

__all__ = ["LLMWrapper", "GPTWrapper", "_fix_json_escapes"]

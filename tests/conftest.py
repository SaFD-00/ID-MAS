"""테스트 설정: GPU/vllm 의존성 없이 테스트 실행을 위한 mock 설정."""
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_LEARNING_LOOP_DIR = os.path.join(_PROJECT_ROOT, "learning_loop")

# learning_loop/__init__.py가 StudentModel 등을 import하면서
# vllm, torch 등 GPU 의존성을 요구합니다.
# 테스트에서는 graph 모듈만 필요하므로 의존성 체인을 우회합니다.

# 1) GPU/ML 의존성 mock
_GPU_MODULES = [
    "vllm",
    "torch",
    "torch.cuda",
    "transformers",
    "accelerate",
]

for mod_name in _GPU_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# 2) models 패키지 mock (vllm에 직접 의존)
_MODEL_MODULES = [
    "models",
    "models.local_model_mixin",
    "models.student_wrapper",
    "models.teacher_wrapper",
    "models.remote_model",
]

for mod_name in _MODEL_MODULES:
    if mod_name not in sys.modules:
        mock_mod = MagicMock()
        mock_mod.__path__ = []
        sys.modules[mod_name] = mock_mod

# 3) learning_loop 패키지를 실제 경로로 등록 (graph 하위 모듈 접근 가능)
if "learning_loop" not in sys.modules:
    ll_module = types.ModuleType("learning_loop")
    ll_module.__path__ = [_LEARNING_LOOP_DIR]
    ll_module.__file__ = os.path.join(_LEARNING_LOOP_DIR, "__init__.py")
    ll_module.StudentModel = MagicMock
    ll_module.TeacherModel = MagicMock
    sys.modules["learning_loop"] = ll_module

# 4) learning_loop.student_model / teacher_model mock
for sub in ["learning_loop.student_model", "learning_loop.teacher_model"]:
    if sub not in sys.modules:
        sys.modules[sub] = MagicMock()

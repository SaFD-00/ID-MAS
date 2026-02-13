"""원격 GPU에서 vLLM 모델을 실행하기 위한 subprocess 기반 프록시 모듈.

다른 GPU에 모델을 격리하여 로드하기 위해 자식 프로세스를 사용합니다.
자식 프로세스는 CUDA 초기화 전에 CUDA_VISIBLE_DEVICES를 설정하므로
GPU 격리가 보장됩니다. 다중 GPU tensor parallel도 지원합니다.

주요 클래스:
    RemoteLLMProxy: 메인 프로세스에서 사용하는 프록시 클래스

사용 예시:
    >>> from models.remote_model import RemoteLLMProxy
    >>> # 단일 GPU
    >>> proxy = RemoteLLMProxy("Qwen/Qwen3-8B", gpu_ids=(1,))
    >>> # 다중 GPU (tensor parallel)
    >>> proxy = RemoteLLMProxy("Qwen/Qwen3-32B", gpu_ids=(0, 1, 2))
    >>> outputs = proxy.chat(messages=[messages], sampling_params=params)
"""
import os
import atexit
import multiprocessing as mp
from typing import Optional, List, Dict, Any, Tuple


class _RemoteCompletionOutput:
    """vLLM CompletionOutput 호환 래퍼."""

    def __init__(self, text: str):
        self.text = text


class _RemoteOutput:
    """vLLM RequestOutput 호환 래퍼."""

    def __init__(self, texts: List[str]):
        self.outputs = [_RemoteCompletionOutput(t) for t in texts]


def _remote_model_worker(
    conn: mp.connection.Connection,
    model_name: str,
    gpu_ids: Tuple[int, ...],
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
):
    """자식 프로세스에서 vLLM 모델을 로드하고 chat 요청을 처리합니다.

    CUDA 초기화 전에 CUDA_VISIBLE_DEVICES를 설정하여 GPU 격리를 보장합니다.
    다중 GPU가 지정되면 tensor parallel로 모델을 로드합니다.

    Args:
        conn: 메인 프로세스와의 Pipe 연결
        model_name: 로드할 모델명
        gpu_ids: 사용할 GPU 인덱스 tuple (예: (0,), (0, 1, 2))
        dtype: 모델 데이터 타입
        tensor_parallel_size: 텐서 병렬 처리 GPU 수
        gpu_memory_utilization: GPU 메모리 활용률
        max_model_len: 최대 시퀀스 길이
    """
    # CUDA 초기화 전에 GPU 설정 (다중 GPU 지원)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    from vllm import LLM

    print(f"[RemoteWorker] Loading model {model_name} on GPU {gpu_ids} (tp={tensor_parallel_size})...")

    llm_kwargs = {
        "model": model_name,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "attention_config": {"backend": "TRITON_ATTN"},
    }

    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)
    print(f"[RemoteWorker] Model loaded: {model_name} on GPU {gpu_ids}")

    # 메인 프로세스에 준비 완료 알림
    conn.send({"status": "ready"})

    # 요청 루프
    while True:
        try:
            request = conn.recv()
        except EOFError:
            break

        if request is None or request.get("command") == "shutdown":
            break

        if request.get("command") == "chat":
            try:
                chat_kwargs = request["kwargs"]
                outputs = llm.chat(**chat_kwargs)
                # 결과를 직렬화 가능한 형태로 변환
                result_texts = []
                for output in outputs:
                    texts = [o.text for o in output.outputs]
                    result_texts.append(texts)
                conn.send({"status": "ok", "result": result_texts})
            except Exception as e:
                conn.send({"status": "error", "error": str(e)})


class RemoteLLMProxy:
    """원격 GPU에서 실행되는 vLLM 모델의 프록시 클래스.

    vLLM LLM.chat()과 동일한 인터페이스를 제공합니다.
    내부적으로 자식 프로세스를 통해 지정된 GPU에서 모델을 실행합니다.
    다중 GPU tensor parallel도 지원합니다.

    Attributes:
        model_name: 모델명
        gpu_ids: GPU 인덱스 tuple
        _process: 자식 프로세스
        _conn: Pipe 연결

    Example:
        >>> proxy = RemoteLLMProxy("Qwen/Qwen3-8B", gpu_ids=(1,))
        >>> proxy = RemoteLLMProxy("Qwen/Qwen3-32B", gpu_ids=(0, 1, 2))
        >>> outputs = proxy.chat(
        ...     messages=[messages],
        ...     sampling_params=params,
        ...     chat_template_kwargs={"enable_thinking": False},
        ... )
        >>> text = outputs[0].outputs[0].text
    """

    def __init__(
        self,
        model_name: str,
        gpu_ids: Tuple[int, ...],
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
    ):
        """RemoteLLMProxy를 초기화합니다.

        자식 프로세스를 spawn하고 모델 로드 완료를 대기합니다.

        Args:
            model_name: 로드할 모델명
            gpu_ids: 사용할 GPU 인덱스 tuple (예: (0,), (0, 1, 2))
            dtype: 모델 데이터 타입
            tensor_parallel_size: 텐서 병렬 처리 GPU 수
            gpu_memory_utilization: GPU 메모리 활용률
            max_model_len: 최대 시퀀스 길이
        """
        self.model_name = model_name
        self.gpu_ids = gpu_ids

        # Pipe 생성 (duplex)
        self._conn, child_conn = mp.Pipe()

        # 자식 프로세스 시작 (daemon=False: vLLM이 내부 자식 프로세스를 생성하므로)
        self._process = mp.Process(
            target=_remote_model_worker,
            args=(
                child_conn,
                model_name,
                gpu_ids,
                dtype,
                tensor_parallel_size,
                gpu_memory_utilization,
                max_model_len,
            ),
            daemon=False,
        )
        self._process.start()

        # 메인 프로세스 종료 시 자식 프로세스 정리
        atexit.register(self.shutdown)

        # 모델 로드 완료 대기
        response = self._conn.recv()
        if response.get("status") != "ready":
            raise RuntimeError(
                f"[RemoteLLMProxy] Failed to load model {model_name} on GPU {gpu_ids}"
            )

        print(f"[RemoteLLMProxy] Model ready: {model_name} on GPU {gpu_ids}")

    def chat(self, **kwargs) -> List[_RemoteOutput]:
        """vLLM chat API와 동일한 인터페이스로 텍스트를 생성합니다.

        Args:
            **kwargs: vLLM LLM.chat()에 전달할 인자
                - messages: 메시지 리스트
                - sampling_params: SamplingParams 인스턴스
                - chat_template_kwargs: 채팅 템플릿 옵션

        Returns:
            _RemoteOutput 리스트 (vLLM RequestOutput 호환)

        Raises:
            RuntimeError: 자식 프로세스에서 오류 발생 시
        """
        self._conn.send({"command": "chat", "kwargs": kwargs})
        response = self._conn.recv()

        if response["status"] == "error":
            raise RuntimeError(
                f"[RemoteLLMProxy] Chat error: {response['error']}"
            )

        # 결과를 vLLM 호환 래퍼로 변환
        return [_RemoteOutput(texts) for texts in response["result"]]

    def __del__(self):
        """프록시 소멸 시 자식 프로세스를 정리합니다."""
        self.shutdown()

    def shutdown(self):
        """자식 프로세스를 종료합니다."""
        try:
            if hasattr(self, '_conn') and self._conn:
                self._conn.send({"command": "shutdown"})
                self._conn.close()
                self._conn = None
        except (BrokenPipeError, OSError):
            pass
        try:
            if hasattr(self, '_process') and self._process and self._process.is_alive():
                self._process.join(timeout=10)
                if self._process.is_alive():
                    self._process.terminate()
        except Exception:
            pass

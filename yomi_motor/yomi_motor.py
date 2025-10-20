# yomi_motor.py
import json
import threading
import os, time
from dotenv import load_dotenv
from openai import OpenAI
import openai
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
# from yomi_motor_run import MotionSequenceExecutor

class EmotionJsonPicker:
    """
    - base_dir 아래 9개 감정 폴더의 *.json을 스캔해서 pools[emotion] = [ {path, name, data}, ... ] 로 보관
    - 각 감정별로 중복 없이 랜덤 pick()
    - 해당 감정의 항목을 전부 1회씩 쓰면 그 감정만 reset (다음 라운드로)
    - init 시 바로 스캔/초기화 수행
    """
    EMOTIONS_DEFAULT = [
        "joy", "sadness", "angry", "fear",
        "surprise", "disgust", "trust", "anticipation", "no"
    ]

    def __init__(
        self,
        base_dir: Optional[str] = None,
        pattern: str = "*.json",
        recursive: bool = False,
        on_key_reset: Optional[Callable[[str], None]] = None,
        seed: Optional[int] = None,
        strict: bool = False,
        encoding: str = "utf-8",
        on_round_complete: Optional[Callable[[str], None]] = None,
        # LLM 설정 (같은 클래스 내부)
        llm_temperature: float = 0.2,
        enable_llm: bool = True,
    ):
        """
        base_dir : 감정 폴더들이 모여있는 루트 디렉토리
        pattern  : JSON 글롭 패턴(기본: *.json)
        recursive: 하위 폴더까지 탐색할지 여부
        on_key_reset: 특정 감정 라운드 종료 시 호출되는 콜백(key 전달)
        seed     : 난수 시드
        strict   : 감정 폴더가 없거나 JSON 파싱 실패 시 예외를 던질지 여부(False면 경고만)
        """
        import random

        if base_dir is None:
            here = Path(__file__).resolve()
            candidates = [
                here.parent / "motion",                                  # 같은 폴더 기준 ../motion
                here.parent.parent / "motion",                            # 두 단계 위 /motion
                here.parent.parent / "yomi_motor" / "motion",             # <repo>/yomi_motor/motion
            ]
            base_dir = next((c for c in candidates if c.exists()), candidates[0])
        self.base_dir = Path(base_dir).resolve()
        self.emotions = self.EMOTIONS_DEFAULT
        self.pattern = pattern
        self.recursive = recursive
        self.on_key_reset = on_key_reset
        self.on_round_complete = on_round_complete
        self.strict = strict
        self.encoding = encoding

        self.rng = random.Random(seed)
        self._lock = threading.RLock()
        
        self._refill_locks: Dict[str, threading.Lock] = {
            emo: threading.Lock() for emo in self.emotions
        }

        # LLM 설정/클라이언트
        self.enable_llm = enable_llm
        self.openai_model = "ft:gpt-4o-2024-08-06:personal::CR1Ggcr5"
        self.llm_temperature = llm_temperature
        self._openai_client = None
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY") 
        print(openai_api_key)
        if self.enable_llm:
            self._init_openai_client(openai_api_key)
        

        # self.motor = MotionSequenceExecutor()
        self.motor = None

        # init 시 바로 스캔 및 상태 초기화
        self._scan_and_build_pools()

    # --------- public API ---------
    def pick(self, key: str) -> Dict[str, Any]:
        """
        해당 감정(key)에서 아직 안 쓴 항목 하나를 무작위로 반환.
        반환값 예시: {"path": ".../joy/happy1.json", "name": "happy1", "data": {...}}
        감정의 모든 항목을 1회씩 쓰면 on_key_reset 콜백 호출 후 해당 감정만 reset.
        """
        self._check_key(key)
        max_attempts = 3
        attempts = 0
        entry: Optional[Dict[str, Any]] = None

        while attempts < max_attempts:
            with self._lock:
                used_row = self.used[key]
                pool_row = self.pools[key]

                if not pool_row:
                    raise RuntimeError(f"[EmotionJsonPicker] '{key}' 폴더에 JSON이 없습니다: {self.base_dir / key}")
                
                # 아직 안 쓴 후보들
                candidates = [i for i, flag in enumerate(used_row) if not flag]
                if candidates:
                    j = self.rng.choice(candidates)
                    used_row[j] = True
                    entry = pool_row[j]
                    print(f"[YOMI_MOTOR] (pick) emotion:{key}, idx:{j}, name:{entry['name']}")
                    break
                
                refill_lock = self._refill_locks[key]
                acquired = refill_lock.acquire(blocking=False)
                try:
                    if acquired:
                        # 알림 콜백
                        if self.on_round_complete:
                            try:
                                self.on_round_complete(key)
                            except Exception as e:
                                print(f"[EmotionJsonPicker] on_round_complete 에러: {e}")

                        # LLM 생성
                        if not self.enable_llm:
                            raise RuntimeError("[EmotionJsonPicker] 라운드 소진되었지만 LLM이 비활성화되어 있습니다.")
                        if not self.openai_model:
                            raise RuntimeError("[EmotionJsonPicker] openai_model이 설정되지 않았습니다.")
                        if not self._openai_client:
                            raise RuntimeError("[EmotionJsonPicker] OpenAI 클라이언트 초기화 실패(키/SDK 확인).")

                        try:
                            new_obj = self._llm_generate_json(key)
                        except Exception as e:
                            print(f"[EmotionJsonPicker] LLM 생성 에러: {e}")
                            # 생성 실패면 더 진행해봤자 후보가 늘지 않으므로 재시도/백오프는 상위에서 정책적으로 처리
                            raise

                        try:
                            final_path = self._next_sequential_filepath(key)
                            self._atomic_dump_json(new_obj, final_path)
                            print(f"[EmotionJsonPicker] 새 후보 저장: {final_path}")
                        except Exception as e:
                            print(f"[EmotionJsonPicker] 새 후보 저장 실패: {e}")
                            # 저장 실패 시 재스캔만으로 후보가 늘지 않으므로 그대로 루프 재시도(결국 예외 가능)

                        # 동기 재스캔(I/O만)
                        self._rescan_key_only(key)
                    else:
                        # 다른 스레드가 리필 중이면 잠깐 대기 후 다시 확인
                        time.sleep(0.01)
                finally:
                    if acquired:
                        refill_lock.release()

            attempts += 1


        # threading.Thread(
        #     target=self.motor.execute_sequence_data,
        #     args=(entry["data"],),
        #     daemon=True
        # ).start()

        return entry

    def remaining(self, key: Optional[str] = None):
        """남은 개수 조회. key=None이면 {emotion: count} dict, 아니면 int."""
        with self._lock:
            if key is None:
                return {k: sum(not f for f in self.used[k]) for k in self.emotions}
            self._check_key(key)
            return sum(not f for f in self.used[key])

    def counts(self) -> Dict[str, int]:
        """초기 스캔 기준 각 감정 폴더의 항목 개수(dict)."""
        with self._lock:
            return {k: len(self.pools[k]) for k in self.emotions}

    def reload(self):
        """디스크를 다시 스캔하여 pools를 갱신하고 사용 상태를 초기화."""
        with self._lock:
            self._scan_and_build_pools()
    
    # --------- internal: LLM ----------
    def _init_openai_client(self, api_key: Optional[str]):
        if not OpenAI:
            print("[EmotionJsonPicker] 경고: OpenAI SDK 임포트 실패. LLM 비활성화됨.")
            self._openai_client = None
            self.enable_llm = False
            return
        # 환경변수 우선
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        try:
            self._openai_client = OpenAI()
        except Exception as e:
            print(f"[EmotionJsonPicker] OpenAI 클라이언트 초기화 실패: {e}")
            self._openai_client = None
            self.enable_llm = False

    def _generate_json_for_key(self, key: str) -> Any:
        """
        LLM이 켜져 있으면 LLM으로 생성, 아니면 더미 JSON 생성.
        """
        if not self.enable_llm:
            return self._dummy_generate_json(key)

        if not self.openai_model or not self._openai_client:
            raise RuntimeError("[EmotionJsonPicker] LLM이 활성화됐지만 모델/클라이언트 준비가 안 됨")

        return self._llm_generate_json(key)

    def _llm_generate_json(self, key: str) -> Any:
        """
        LLM을 호출해 JSON-serializable 객체를 반환.
        모델은 self.openai_model 사용, 프롬프트는 감정 key 하나로 단순화.
        """
        client = self._openai_client
        resp = client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "user", "content": key},
            ],
            temperature=self.llm_temperature,
        )
        text = resp.choices[0].message.content
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 모델이 JSON만 반환하지 못하는 경우 대비
            print(f"[EmotionJsonPicker] LLM이 유효하지 않은 JSON을 반환했습니다: {text}")
            raise RuntimeError(f"LLM이 '{key}'에 대해 유효한 JSON을 반환하지 못했습니다.") from e

    def _dummy_generate_json(self, key: str) -> Any:
        """LLM 비활성 시 사용할 더미 생성기(테스트용)."""
        return {
            "emotion": key,
            "generated_at": int(time.time()),
            "sequence": [
                {"act": "start", "duration": 0.2},
                {"act": f"{key}_pose", "duration": 1.0},
                {"act": "end", "duration": 0.2},
            ],
        }


    # --------- internal ---------
    def _scan_and_build_pools(self):
        """디스크에서 JSON을 스캔해 pools/used를 만든다."""
        pools: Dict[str, List[Dict[str, Any]]] = {}
        for emo in self.emotions:
            entries = self._scan_one_emotion(emo)
            pools[emo] = entries
        self.pools = pools
        self.used = {k: [False] * len(v) for k, v in pools.items()}
        
    def _scan_one_emotion(self, emo: str) -> List[Dict[str, Any]]:
        d = self.base_dir / emo
        if not d.exists() or not d.is_dir():
            msg = f"[EmotionJsonPicker] 경고: '{emo}' 폴더가 없습니다: {d}"
            if self.strict:
                raise FileNotFoundError(msg)
            print(msg)
            return []
        
        it = d.rglob(self.pattern) if self.recursive else d.glob(self.pattern)
        files = sorted(it)
        entries: List[Dict[str, Any]] = []
        for f in files:
            try:
                with open(f, "r", encoding=self.encoding) as fh:
                    data = json.load(fh)
                entries.append({
                    "path": str(f.resolve()),
                    "name": f.stem,
                    "data": data,
                })
            except Exception as e:
                msg = f"[EmotionJsonPicker] JSON 로드 실패: {f} ({e})"
                if self.strict:
                    raise RuntimeError(msg)
                print(msg)
        return entries

    def _rescan_key_only(self, key: str):
        """해당 감정 폴더만 다시 스캔하여 pools[key], used[key] 갱신 (생성 없음)."""
        self._check_key(key)
        new_entries = self._scan_one_emotion(key)
        with self._lock:
            self.pools[key] = new_entries
            self.used[key] = [False] * len(new_entries)

    def _next_sequential_filepath(self, key: str) -> Path:
        """
        key(\d+).json 중 최대 번호 + 1 로 파일 경로를 생성.
        예: joy0..joy5.json 있으면 joy6.json 반환.
        """
        d = self.base_dir / key
        d.mkdir(parents=True, exist_ok=True)

        it = d.rglob(self.pattern) if self.recursive else d.glob(self.pattern)
        maxn = -1
        prefix = key
        for f in it:
            stem = f.stem
            if stem.startswith(prefix):
                tail = stem[len(prefix):]
                if tail.isdigit():
                    try:
                        n = int(tail)
                        if n > maxn:
                            maxn = n
                    except ValueError:
                        pass
        nextn = maxn + 1
        return d / f"{prefix}{nextn}.json"
    
    def _atomic_dump_json(self, obj: Any, final_path: Path):
        """JSON을 임시파일에 쓴 뒤 os.replace로 원자적 저장."""
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding=self.encoding, dir=str(final_path.parent)
        ) as tmp:
            json.dump(obj, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, final_path)

    def _check_key(self, key: str):
        if not hasattr(self, "pools") or key not in self.pools:
            raise KeyError(f"알 수 없는 감정 키: {key!r}. 사용 가능: {list(getattr(self, 'pools', {}).keys())}")

    # def _reset_key_internal(self, key: str):
    #     """해당 감정만 새 라운드 시작(used를 False로 초기화)"""
    #     with self._lock:
    #         self._check_key(key)
    #         self.used[key] = [False] * len(self.pools[key])
        
    #     def _worker():
    #         try:
    #             if self.on_round_complete:
    #                 self.on_round_complete(key)   # 외부 함수 호출
    #         except Exception as e:
    #             print(f"[EmotionJsonPicker] on_round_complete 에러: {e}")
    #         # 해당 감정만 다시 스캔
    #         self._rescan_key(key)
    #     threading.Thread(target=_worker, daemon=True).start()
    
    # def _rescan_key(self, key: str):
    #     """해당 감정 폴더만 다시 스캔하여 pools[key], used[key] 갱신.
    #     다른 감정의 used/pools는 유지."""
    #     self._check_key(key)

    #    # [1. 파일명 결정을 위해 현재 파일 개수 스캔]
    #     # (API 호출 전에 파일명을 미리 정해야 함)
        
    #     # 타겟 디렉토리 경로 설정
    #     d = self.base_dir / key
        
    #     # 폴더가 없으면 생성 (파일 개수 세기 전에 필요)
    #     d.mkdir(parents=True, exist_ok=True) 

    #     try:
    #         # 현재 폴더의 .json 파일 개수를 셉니다.
    #         it = d.rglob(self.pattern) if self.recursive else d.glob(self.pattern)
    #         current_count = len(list(it))
    #         new_file_number = current_count + 1
            
    #         # [핵심 수정] 새 파일명 결정
    #         filename = f"{key}{new_file_number}.json"
    #         filepath = d / filename
            
    #     except Exception as e:
    #         # 만약 파일 개수 세기에 실패하면 (예: 권한 문제) 타임스탬프로 대체
    #         print(f"[EmotionJsonPicker] 파일 개수 세기 실패: {e}. 타임스탬프 이름으로 대체합니다.")
    #         timestamp = int(time.time())
    #         filename = f"openai_gen_{key}_{timestamp}.json"
    #         filepath = d / filename


    #     # [2. API 호출 및 파일 저장 로직]
    #     try:
    #         if not self.openai_client:
    #             raise Exception("OpenAI 클라이언트가 초기화되지 않았습니다. (API 키 확인 필요)")

    #         print(f"[EmotionJsonPicker] '{key}' 감정의 새 모션을 OpenAI로 생성합니다...")
    #         response = self.openai_client.chat.completions.create(
    #             model="ft:gpt-4o-2024-08-06:personal::CR1Ggcr5", # 사용자 파인튜닝 모델
    #             messages=[
    #                 {"role": "user", "content": key}
    #             ]
    #         )
    #         output_text = response.choices[0].message.content

    #         # 응답 파싱
    #         try:
    #             output_json = json.loads(output_text)
    #         except json.JSONDecodeError:
    #             print(f"[EmotionJsonPicker] '{key}' 응답이 JSON 형식이 아닙니다. 원본을 저장합니다.")
    #             output_json = {"raw_output": output_text, "error": "Not valid JSON"}

    #         # [핵심 수정] 위에서 계산된 'filepath' (예: .../joy/joy3.json)로 저장합니다.
    #         with open(filepath, "w", encoding="utf-8") as f:
    #             json.dump(output_json, f, indent=2, ensure_ascii=False)
            
    #         print(f"[EmotionJsonPicker] 새 모션 저장 완료: {filepath}")

    #     except Exception as e:
    #         # API 호출이나 파일 저장이 실패해도 리스캔은 시도해야 함
    #         print(f"[EmotionJsonPicker] OpenAI API 호출 또는 파일 저장 실패: {e}")

        
    #     # [3. 폴더 리스캔 로직] (기존과 동일)
    #     # (API 호출 성공 여부와 관계없이 폴더 전체를 다시 읽어들임)
        
    #     if not d.exists() or not d.is_dir():
    #         if self.strict:
    #             raise FileNotFoundError(f"[EmotionJsonPicker] '{key}' 폴더가 없습니다: {d}")
    #         print(f"[EmotionJsonPicker] 경고: '{key}' 폴더가 없습니다: {d}")
    #         new_entries: List[Dict[str, Any]] = []
    #     else:
    #         # 이제 이 glob은 방금 생성된 'keyN.json' 파일도 포함합니다.
    #         it = d.rglob(self.pattern) if self.recursive else d.glob(self.pattern)
    #         files = sorted(it)
    #         new_entries: List[Dict[str, Any]] = []
    #         for f in files:
    #             try:
    #                 with open(f, "r", encoding=self.encoding) as fh:
    #                     data = json.load(fh)
    #                 new_entries.append({"path": str(f.resolve()), "name": f.stem, "data": data})
    #             except Exception as e:
    #                 msg = f"[EmotionJsonPicker] JSON 로드 실패: {f} ({e})"
    #                 if self.strict:
    #                     raise RuntimeError(msg)
    #                 print(msg)

    #     # [4. 데이터 교체] (기존과 동일)
    #     # 실제 교체는 짧게 락(lock)을 잡고 수행
    #     with self._lock:
    #         # self._check_key(key) # _rescan_key 맨 위에서 이미 검사함
    #         self.pools[key] = new_entries
    #         self.used[key]  = [False] * len(new_entries)  # 해당 감정만 초기화

    # def _check_key(self, key: str):
    #     if key not in self.pools:
    #         raise KeyError(f"알 수 없는 감정 키: {key!r}. 사용 가능: {list(self.pools.keys())}")

# --------- 테스트 코드 ---------
if __name__ == "__main__":
    import random, time

    def on_round_complete(key: str):
        print(f"🔄 round complete for '{key}' → rescanning...")

    # 필요에 따라 base_dir/recursive 수정
    picker = EmotionJsonPicker(
        base_dir=None,          # None이면 자동 추정
        recursive=False,        # 하위 폴더까지 포함하려면 True
        seed=42,                # 재현성 있는 랜덤
        on_round_complete=on_round_complete,
    )

    print("▶ initial counts:", picker.counts())

    emotions = picker.emotions  # ["joy", "sadness", ... , "no"]
    for t in range(20):
        key = random.choice(emotions)
        try:
            entry = picker.pick(key)  # {'path','name','data'}
            steps = len(entry["data"]) if isinstance(entry.get("data"), list) else "?"
            print(f"[{t+1:02d}] key={key:<13} → name={entry['name']:<20} steps={steps}")
        except RuntimeError as e:
            # 폴더 비었을 때 등
            print(f"[{t+1:02d}] key={key:<13} → SKIP ({e})")
        time.sleep(0.05)

    print("▶ remaining after 20 picks:", picker.remaining())
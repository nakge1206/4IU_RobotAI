# yomi_motor.py
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from yomi_motor_run import MotionSequenceExecutor

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
        self._lock = threading.Lock()

        self.motor = MotionSequenceExecutor()

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
        with self._lock:
            used_row = self.used[key]
            pool_row = self.pools[key]

            if not pool_row:
                raise RuntimeError(f"[EmotionJsonPicker] '{key}' 폴더에 JSON이 없습니다: {self.base_dir / key}")
            
            # 아직 안 쓴 후보들
            candidates = [i for i, flag in enumerate(used_row) if not flag]

            # (방어) 후보가 없다면 해당 key만 새 라운드 시작
            if not candidates:
                self._reset_key_internal(key)
                used_row = self.used[key]
                pool_row = self.pools[key]
                candidates = list(range(len(pool_row)))

            j = self.rng.choice(candidates)
            used_row[j] = True
            entry = pool_row[j]
            round_done = all(used_row)
            print(f"[YOMI_MOTOR] (pick) 감정:{key}, idx:{j}")

        if round_done:
            self._reset_key_internal(key)


        threading.Thread(
            target=self.motor.execute_sequence_data,
            args=(entry["data"],),
            daemon=True
        ).start()

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

    # --------- internal ---------
    def _scan_and_build_pools(self):
        """디스크에서 JSON을 스캔해 pools/used를 만든다."""
        pools: Dict[str, List[Dict[str, Any]]] = {}
        for emo in self.emotions:
            d = self.base_dir / emo
            if not d.exists() or not d.is_dir():
                msg = f"[EmotionJsonPicker] 경고: '{emo}' 폴더가 없습니다: {d}"
                if self.strict:
                    raise FileNotFoundError(msg)
                print(msg)
                pools[emo] = []
                continue

            it = d.rglob(self.pattern) if self.recursive else d.glob(self.pattern)
            files = sorted(it)  # 재현성 위해 정렬
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
            pools[emo] = entries

        self.pools = pools
        self.used = {k: [False] * len(v) for k, v in pools.items()}

    def _reset_key_internal(self, key: str):
        """해당 감정만 새 라운드 시작(used를 False로 초기화)"""
        with self._lock:
            self._check_key(key)
            self.used[key] = [False] * len(self.pools[key])
        
        def _worker():
            try:
                if self.on_round_complete:
                    self.on_round_complete(key)   # 외부 함수 호출
            except Exception as e:
                print(f"[EmotionJsonPicker] on_round_complete 에러: {e}")
            # 해당 감정만 다시 스캔
            self._rescan_key(key)
        threading.Thread(target=_worker, daemon=True).start()
    
    def _rescan_key(self, key: str):
        """해당 감정 폴더만 다시 스캔하여 pools[key], used[key] 갱신.
        다른 감정의 used/pools는 유지."""
        self._check_key(key)
        d = self.base_dir / key
        if not d.exists() or not d.is_dir():
            if self.strict:
                raise FileNotFoundError(f"[EmotionJsonPicker] '{key}' 폴더가 없습니다: {d}")
            print(f"[EmotionJsonPicker] 경고: '{key}' 폴더가 없습니다: {d}")
            new_entries: List[Dict[str, Any]] = []
        else:
            it = d.rglob(self.pattern) if self.recursive else d.glob(self.pattern)
            files = sorted(it)
            new_entries: List[Dict[str, Any]] = []
            for f in files:
                try:
                    with open(f, "r", encoding=self.encoding) as fh:
                        data = json.load(fh)
                    new_entries.append({"path": str(f.resolve()), "name": f.stem, "data": data})
                except Exception as e:
                    msg = f"[EmotionJsonPicker] JSON 로드 실패: {f} ({e})"
                    if self.strict:
                        raise RuntimeError(msg)
                    print(msg)

        # 실제 교체는 짧게 락 잡고 수행
        with self._lock:
            self.pools[key] = new_entries
            self.used[key]  = [False] * len(new_entries)  # 해당 감정만 초기화

    def _check_key(self, key: str):
        if key not in self.pools:
            raise KeyError(f"알 수 없는 감정 키: {key!r}. 사용 가능: {list(self.pools.keys())}")

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
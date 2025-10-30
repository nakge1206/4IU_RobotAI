import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os
import re
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import networkx as nx


class MemoryGraph:
    """ 사건/대화 내용을 그래프 구조로 저장하는 클래스 """

    def __init__(self, path="memory_graph"):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.graph_file = os.path.join(path, "memory_graph.json")
        self.graph = nx.DiGraph()
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        if os.path.exists(self.graph_file):
            self.load()

    def add_event(self, text):
        """ 대화 이벤트를 노드로 추가 """
        emb = self.encoder.encode([text])[0].tolist()
        node_id = f"event_{len(self.graph.nodes)}"
        self.graph.add_node(node_id, text=text, embedding=emb, time=time.time())
        if len(self.graph.nodes) > 1:
            prev_id = f"event_{len(self.graph.nodes)-2}"
            self.graph.add_edge(prev_id, node_id)
        self.save()

    def search_related(self, query, topk=3):
        """ 쿼리와 유사한 이벤트 검색 """
        if not self.graph.nodes:
            return []
        q_emb = self.encoder.encode([query])[0]
        scored = []
        for nid, data in self.graph.nodes(data=True):
            emb = torch.tensor(data["embedding"])
            sim = torch.nn.functional.cosine_similarity(
                torch.tensor(q_emb), emb, dim=0
            ).item()
            scored.append((sim, data["text"]))
        scored.sort(reverse=True)
        return [t for _, t in scored[:topk]]

    def save(self):
        data = {
            "nodes": [
                (nid, self.graph.nodes[nid]) for nid in self.graph.nodes
            ],
            "edges": list(self.graph.edges)
        }
        with open(self.graph_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        with open(self.graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(data["nodes"])
        self.graph.add_edges_from(data["edges"])


class LLMResponder:
    def __init__(self,
                 model_path="skt/A.X-4.0-Light",   # ✅ 베이스 모델
                 adapter_path=r"C:/Users/COM/Desktop/yomi/4IU_RobotAI/yomi_core/llm_core/ax4_lora_finetune_20251002_154704/checkpoint-2495"  # ✅ LoRA
                 ):
        print("[LLMResponder] 초기화 중...")

        # 베이스 토크나이저/모델은 HF 원본에서 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            # local_files_only=True,  # 캐시에 100% 있을 때만 켜세요
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            # local_files_only=True,  # 캐시에 100% 있을 때만 켜세요
        )

        # LoRA 어댑터 연결
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            local_files_only=True   # 어댑터는 로컬이 맞음
        )

        self.model.eval()
        print("[LLMResponder] 모델 준비 완료")


        # 그래프 기반 메모리 추가
        self.memory_graph = MemoryGraph()

        # 캐릭터 초기 정보
        self.character_info = (
            "이 캐릭터는 5살 여자아이 요미야. 이름은 요미이고, MBTI는 ESTJ야. "
            "요미는 유치원에 다니지만 말을 잘 듣지 않고, 종종 장난을 치는 아이야. "
            "대화와 경험을 통해 점점 바른 행동을 배워. "
            "요미는 때때로 투정부리거나 짜증을 내지만, 대화를 통해 반성하고 배우는 태도를 보여야 해."
        )

    def wrap_prompt(self, text, emotion=None, event=None, mbti=None, vision=None):

        # ---- 가중치 자동 결정 ----
        has_text = bool(text and text.strip())
        has_vision = bool(vision and str(vision).strip())

        if has_text and has_vision:
            text_weight  = self.modality_weights.get("speech", 0.7)
            vision_weight = self.modality_weights.get("vision", 0.3)
        elif has_text and not has_vision:
            text_weight, vision_weight = 1.0, 0.0
        elif has_vision and not has_text:
            text_weight, vision_weight = 0.0, 1.0
        else:
            # 둘 다 없음: 안전한 기본 프롬프트
            text_weight, vision_weight = 1.0, 0.0

        meta_info = []
        if emotion:
            meta_info.append(f"사용자 감정은 '{emotion}'이야.")
        if event:
            meta_info.append(f"상황은 '{event}'이야.")
        if mbti:
            meta_info.append(f"사용자 MBTI는 '{mbti}'야.")
        if vision:
            meta_info.append(f"시각 정보: {vision}")

        # 그래프에서 관련 기억 꺼내오기
        related = self.memory_graph.search_related(text, topk=3)
        related_str = "\n".join([f"- {r}" for r in related]) if related else "없음"

        meta_str = " ".join(meta_info) if meta_info else "정보 없음"

        # ---- 가중치 표기 + 입력 블록 ----
        # 비어있는 입력은 '없음'으로 표기
        in_text  = text if has_text else "없음"
        in_vision = str(vision) if has_vision else "없음"

        weighted_input = (
            f"[입력 비중: 텍스트 {text_weight*100:.0f}%, 비전 {vision_weight*100:.0f}%]\n"
            f"텍스트 내용: {in_text}\n"
            f"비전 내용: {in_vision}\n"
        )
            


        return (
            f"### 캐릭터 요약\n{self.character_info}\n\n"
            f"### 사용자 정보\n{meta_str}\n\n"
            f"### 과거 관련 기억\n{related_str}\n\n"
            "### 시스템 지침\n"
            "너는 유아야. 입력 내용을 보고 상황에 맞는 유아 말투의 자연스러운 반응을 해줘.\n"
            "그리고 반드시 감정을 함께 추론해. 감정은 다음 8가지 중 하나만 선택해:\n"
            "- 기쁨, 신뢰, 공포, 놀람, 슬픔, 혐오, 분노, 기대\n"
            "감정이 불명확하면 기본값은 '기쁨'이야.\n\n"
            "출력은 아래 형식을 따라야 해:\n"
            "예시:\n"
            "\"대답\": 안녕! 난 요미야!\n"
            "\"감정\": 기쁨\n\n"
            f"### 입력 정보\n{weighted_input}\n"
            "### 출력:"
        )

    def generate_response(self, text, emotion=None, event=None, mbti=None):
        prompt = self.wrap_prompt(text, emotion=emotion, event=event, mbti=mbti)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
        end = time.time()
        print(f"[LLMResponder] 추론 시간: {end - start:.2f}초")

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):].strip()

        print("[🧪 모델 출력 원문]", repr(output_text))

        # ───────────── 응답 후처리 (정규식 기반) ─────────────
        response_text, emotion_text = None, None

        match_resp = re.search(r'"대답"\s*:\s*["“”]?([^"\n]+)', output_text)
        match_emot = re.search(r'"감정"\s*:\s*["“”]?([^"\n]+)', output_text)

        if match_resp:
            response_text = match_resp.group(1).strip()
        if match_emot:
            emotion_text = match_emot.group(1).strip()

        if not response_text:
            response_text = "못들었어 다시 말해줘!"
        if not emotion_text:
            emotion_text = "기쁨"

        final = f'"대답": {response_text}\n"감정": {emotion_text}'

        # ───────────── 메모리에 저장 ─────────────
        self.memory_graph.add_event(f"user: {text}")
        self.memory_graph.add_event(f"yomi: {response_text} ({emotion_text})")

        return final

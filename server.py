from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
import json
import os
import re

app = FastAPI()

# CORS設定: フロントエンドからの通信を許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_index():
    return FileResponse("index.html")

# 🌟 Gemini APIの設定
# 環境変数 GEMINI_API_KEY があればそれを使用し、なければ直接指定したキーを使用します
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyA_jyfC7rhnA1x1h6y6YqP5VX1IcHVOOMc")
genai.configure(api_key=API_KEY)

# --- データモデル定義 ---
class IdeaRequest(BaseModel):
    persona: str
    target: str
    action: str

class ChatMessage(BaseModel):
    role: str # "user" または "ai"
    text: str

class ChatRequest(BaseModel):
    persona: str
    idea_context: str # タイトルや詳細の要約
    history: list[ChatMessage] # 会話履歴

# 🌟 モデル名のキャッシュ変数
_CACHED_MODEL_NAME = None

def get_gemini_model():
    """利用可能なモデルを自動選択し、キャッシュする関数"""
    global _CACHED_MODEL_NAME
    if _CACHED_MODEL_NAME is not None:
        return genai.GenerativeModel(_CACHED_MODEL_NAME)

    valid_model_name = None
    try:
        # 利用可能なモデルの一覧を取得して、generateContentをサポートしているものを探す
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                valid_model_name = m.name.replace('models/', '')
                # flashモデルがあれば優先的に選択
                if 'flash' in valid_model_name:
                    break
        
        _CACHED_MODEL_NAME = valid_model_name or 'gemini-1.5-flash'
        print(f"🤖 使用モデルを選択しました: {_CACHED_MODEL_NAME}")
        return genai.GenerativeModel(_CACHED_MODEL_NAME)
    except Exception as e:
        print(f"🚨 モデル一覧の取得に失敗しました: {e}")
        # 失敗した場合は標準的な名前をデフォルトとする
        return genai.GenerativeModel('gemini-1.5-flash')

@app.post("/api/generate")
async def generate_idea(req: IdeaRequest):
    """キーワードからアイデアを生成するエンドポイント"""
    # 🧠 非エンジニア向けに専門用語を排除したプロンプト
    prompt = f"""
    あなたは「{req.persona}」です。その視点で、以下のキーワードを掛け合わせた斬新なWebサービスのアイデアを提案してください。
    
    ターゲット: {req.target}
    仕組み: {req.action}
    
    【重要ルール】
    1. アイデアを形にするための「構築ワークフロー」は、専門用語（API、データベース、デプロイ、フロントエンド等）を一切使わないでください。
    2. 現代のAIツール（Cursor, v0, Bolt.newなど）を使えば、プログラミングができなくても「魔法のように作れる」ことを強調した、具体的で簡単な3ステップにしてください。
    3. 「AI完結難易度」は、アイデア次第で誰でも今日中に作れることをポジティブに伝えてください。
    
    出力は必ず以下のJSONフォーマットのみにしてください。余計な解説は不要です。
    {{
        "title": "サービスのタイトル",
        "pitch": "一言で言うと？",
        "details": "なぜ面白いのかの詳細（ペルソナの口調で）",
        "ai_feasibility": "【AIで作れる度: ★☆☆〜★★★】初心者でもAIに〇〇と言えば作れるよ！という解説",
        "workflow": [
            "1. 〇〇（AIツール名）に「〇〇な画面を作って」と頼む",
            "2. AIに〇〇を追加してもらう",
            "3. 〇〇ボタンを押してネットに公開する"
        ],
        "tags": ["タグ1", "タグ2"]
    }}
    """
    try:
        model = get_gemini_model()
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # AIがマークダウン形式(```json ... ```)で返してきた場合に中身を抽出する
        if "```" in result_text:
            match = re.search(r'`{3}(?:json)?\s*(.*?)\s*`{3}', result_text, re.DOTALL)
            if match:
                result_text = match.group(1)
        
        # 不要な制御文字などを掃除してJSONとして読み込む
        data = json.loads(result_text.strip())
        
        # データが不足している場合のフォールバック
        if "tags" not in data: data["tags"] = ["新規アイデア"]
        if "workflow" not in data: data["workflow"] = ["AIツールを開く", "指示を出す", "公開する"]
        
        return data
    except Exception as e:
        print(f"🚨 生成エラー: {str(e)}")
        # パースに失敗した場合はエラーメッセージを返す
        return {"error": "AIの回答を解析できませんでした。もう一度試してみてください。"}

@app.post("/api/chat")
async def chat_idea(req: ChatRequest):
    """生成されたアイデアについて対話（壁打ち）するエンドポイント"""
    prompt = f"""
    あなたは「{req.persona}」です。以下のアイデアについて、プログラミングができないユーザーと一緒に作戦会議（壁打ち）をしています。
    キャラクターになりきって、具体的で簡単なアドバイスをしてください。専門用語は禁止です。
    
    【対象のアイデア】
    {req.idea_context}
    
    【これまでの会話】
    """
    for msg in req.history:
        speaker = "ユーザー" if msg.role == "user" else f"あなた（{req.persona}）"
        prompt += f"{speaker}: {msg.text}\n"
    
    prompt += f"\n返答は「{req.persona}」らしい口調で、相手が「自分でも作れる！」とワクワクするような内容にしてください。"

    try:
        model = get_gemini_model()
        response = model.generate_content(prompt)
        return {"reply": response.text.strip()}
    except Exception as e:
        print(f"🚨 チャットエラー: {str(e)}")
        return {"error": "返信に失敗しました。"}
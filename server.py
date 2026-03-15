from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
import json
import os
import re

app = FastAPI()

# CORS設定（画面側からの通信を許可する設定）
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

# --- データモデル定義 ---
class IdeaRequest(BaseModel):
    persona: str
    target: str
    action: str

class ChatMessage(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    persona: str
    idea_context: str
    history: list[ChatMessage]

@app.post("/api/generate")
async def generate_idea(req: IdeaRequest):
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
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"error": "APIキーが設定されていません。Vercelの環境変数を確認してください。"}
            
        # 最新のSDKでの呼び出し方
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash', # 最新の高速モデル
            contents=prompt,
        )
        result_text = response.text.strip()
        
        # AIがマークダウン形式で返してきた場合に中身を抽出
        if "```" in result_text:
            match = re.search(r'`{3}(?:json)?\s*(.*?)\s*`{3}', result_text, re.DOTALL)
            if match:
                result_text = match.group(1)
        
        # 🌟 確実なJSON抽出：波括弧 { } で囲まれた部分だけを強制的に抜き出す（余計な挨拶文字対策）
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            result_text = result_text[start_idx:end_idx+1]
        
        data = json.loads(result_text)
        
        # データが不足している場合のフォールバック（保険）
        if "tags" not in data: data["tags"] = ["新規アイデア"]
        if "workflow" not in data: data["workflow"] = ["AIツールを開く", "指示を出す", "公開する"]
        if "ai_feasibility" not in data: data["ai_feasibility"] = "【AI完結難易度: 判定不能】AIツールを使えば簡単に作れるはずです！"
        
        return data
    except Exception as e:
        print(f"🚨 生成エラー: {str(e)}")
        return {"error": "AIの回答を解析できませんでした。もう一度お試しください。"}

@app.post("/api/chat")
async def chat_idea(req: ChatRequest):
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
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"error": "APIキーが設定されていません。"}
            
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return {"reply": response.text.strip()}
    except Exception as e:
        print(f"🚨 チャットエラー: {str(e)}")
        return {"error": "返信に失敗しました。"}
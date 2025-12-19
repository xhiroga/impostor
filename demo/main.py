from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"


def _resolve_default_video() -> str:
    """data配下のmp4のうち最初の1本を返す。"""
    videos = sorted(DATA_DIR.glob("*.mp4"))
    if not videos:
        raise HTTPException(status_code=500, detail="動画ファイルが見つかりません")
    return videos[0].name


app = FastAPI()
app.mount("/assets", StaticFiles(directory=DATA_DIR), name="assets")


@app.get("/")
async def read_root():
    video_name = _resolve_default_video()
    return HTMLResponse(
        f"""
        <!DOCTYPE html>
        <html lang=\"ja\">
            <head>
                <meta charset=\"utf-8\" />
                <title>FastAPI + HTMX Demo</title>
                <script src=\"https://unpkg.com/htmx.org@1.9.10\"></script>
                <style>
                    body {{ font-family: system-ui, sans-serif; margin: 0; padding: 2rem; background: #111; color: #f7f7f7; }}
                    main {{ max-width: 960px; margin: 0 auto; }}
                    video {{ width: 100%; height: auto; border-radius: 12px; box-shadow: 0 16px 48px rgba(0,0,0,0.4); }}
                    .panel {{ margin-bottom: 1.5rem; }}
                </style>
            </head>
            <body>
                <main>
                    <h1>FastAPI × HTMX</h1>
                    <section class=\"panel\">
                        <p>dataディレクトリのサンプル動画をそのまま再生しています。</p>
                        <video controls autoplay loop muted playsinline>
                            <source src=\"/assets/{video_name}\" type=\"video/mp4\" />
                            お使いのブラウザは動画再生に対応していません。
                        </video>
                    </section>
                    <button hx-get=\"/api/data\" hx-target=\"#result\" hx-swap=\"innerHTML\">
                        データ取得
                    </button>
                    <div id=\"result\" style=\"margin-top: 1rem;\"></div>
                </main>
            </body>
        </html>
        """
    )


@app.get("/api/data", response_class=HTMLResponse)
async def fetch_data():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"<p>最新データ: {now}</p>"

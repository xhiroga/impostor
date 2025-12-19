from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"


def _list_videos() -> list[Path]:
    videos = sorted(DATA_DIR.glob("*.mp4"))
    if not videos:
        raise HTTPException(status_code=500, detail="動画ファイルが見つかりません")
    return videos


def _resolve_video_name(video: str | None) -> str:
    videos = _list_videos()
    if video is None:
        return videos[0].name
    candidate = DATA_DIR / video
    if candidate.exists() and candidate.suffix == ".mp4" and candidate.parent == DATA_DIR:
        return candidate.name
    raise HTTPException(status_code=404, detail="指定された動画が存在しません")


def _render_video_html(video_name: str) -> str:
    return (
        "<video controls autoplay loop muted playsinline>"
        f"<source src=\"/assets/{video_name}\" type=\"video/mp4\" />"
        "お使いのブラウザは動画再生に対応していません。"
        "</video>"
    )


app = FastAPI()
app.mount("/assets", StaticFiles(directory=DATA_DIR), name="assets")


@app.get("/")
async def read_root():
    videos = _list_videos()
    video_name = videos[0].name
    select_options = "".join(
        f"<option value=\"{video.name}\" {'selected' if video.name == video_name else ''}>{video.stem}</option>"
        for video in videos
    )
    player_html = _render_video_html(video_name)
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
                        <p>dataディレクトリのサンプル動画を選択できます。</p>
                        <label style=\"display:block; margin-bottom:0.5rem;\">
                            動画セレクタ
                            <select
                                name=\"video\"
                                hx-get=\"/player\"
                                hx-trigger=\"change\"
                                hx-target=\"#player\"
                                hx-swap=\"innerHTML\"
                                style=\"margin-left:0.5rem; padding:0.25rem 0.5rem;\"
                            >
                                {select_options}
                            </select>
                        </label>
                        <div id=\"player\">
                            {player_html}
                        </div>
                    </section>
                </main>
            </body>
        </html>
        """
    )


@app.get("/player", response_class=HTMLResponse)
async def render_player(video: str | None = None):
    video_name = _resolve_video_name(video)
    return _render_video_html(video_name)

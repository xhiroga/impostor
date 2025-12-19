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


app = FastAPI()
app.mount("/assets", StaticFiles(directory=DATA_DIR), name="assets")


@app.get("/")
async def read_root():
    videos = _list_videos()
    video_name = videos[0].name
    options = [
        {"value": video.name, "label": video.stem, "selected": video.name == video_name}
        for video in videos
    ]
    select_options = "".join(
        f"<option value=\"{opt['value']}\" {'selected' if opt['selected'] else ''}>{opt['label']}</option>"
        for opt in options
    )
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
                    #viewer {{ width: 100%; aspect-ratio: 16 / 9; background: #fff; border-radius: 16px; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1); position: relative; }}
                    #viewer canvas {{ width: 100%; height: 100%; display: block; border-radius: 16px; }}
                    #status {{ margin-top: 0.5rem; font-size: 0.9rem; color: #ccc; }}
                    #angles {{ margin-top: 0.25rem; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.9rem; color: #bbb; }}
                </style>
            </head>
            <body>
                <main>
                    <h1>FastAPI × HTMX</h1>
                    <section class=\"panel\">
                        <p>ドラッグでカメラを動かすと、角度に応じて81フレームのどれかを自動選択し、キューブに貼り付けます。</p>
                        <label style=\"display:block; margin-bottom:0.75rem;\">
                            動画セレクタ
                            <select id=\"video-select\" style=\"margin-left:0.5rem; padding:0.25rem 0.5rem;\">
                                {select_options}
                            </select>
                        </label>
                        <div id=\"viewer\"></div>
                        <p id=\"status\">読み込み中...</p>
                        <p id=\"angles\"></p>
                    </section>
                </main>
                <script type=\"module\">
                    import * as THREE from 'https://unpkg.com/three@0.169.0/build/three.module.js';
                    import {{ OrbitControls }} from 'https://unpkg.com/three@0.169.0/examples/jsm/controls/OrbitControls.js';

                    const FRAME_COUNT = 81;
                    const PITCH_BANDS = [
                        {{ min: 0, max: 18, offset: 0 }},
                        {{ min: 18, max: 36, offset: 16 }},
                        {{ min: 36, max: 54, offset: 32 }},
                        {{ min: 54, max: 72, offset: 48 }},
                        {{ min: 72, max: 91, offset: 64 }},
                    ];

                    const viewerEl = document.getElementById('viewer');
                    const selectEl = document.getElementById('video-select');
                    const statusEl = document.getElementById('status');
                    const anglesEl = document.getElementById('angles');

                    let renderer, scene, camera, controls, mesh, texture, textureCanvas, textureCtx;
                    let frameBitmaps = [];
                    let loading = false;

                    async function init() {{
                        await loadVideo(selectEl.value);
                        initThree();
                        animate();
                    }}

                    selectEl.addEventListener('change', async () => {{
                        await loadVideo(selectEl.value);
                    }});

                    function updateStatus(text) {{
                        statusEl.textContent = text;
                    }}

                    async function loadVideo(videoName) {{
                        if (!videoName || loading) return;
                        loading = true;
                        updateStatus(`Loading ${{videoName}} ...`);
                        try {{
                            await cleanupFrames();
                            const video = document.createElement('video');
                            video.src = `/assets/${{videoName}}`;
                            video.crossOrigin = 'anonymous';
                            video.muted = true;
                            video.playsInline = true;
                            video.preload = 'auto';
                            await waitEvent(video, 'loadedmetadata');

                            const captureCanvas = document.createElement('canvas');
                            captureCanvas.width = video.videoWidth;
                            captureCanvas.height = video.videoHeight;
                            const captureCtx = captureCanvas.getContext('2d');
                            const duration = video.duration;
                            const maxTime = duration - 0.0001;
                            const frameTimes = Array.from({{ length: FRAME_COUNT }}, (_, i) => Math.min((i / (FRAME_COUNT - 1)) * duration, maxTime));
                            frameBitmaps = [];
                            for (const time of frameTimes) {{
                                await seekVideo(video, time);
                                captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
                                const bitmap = await createImageBitmap(captureCanvas);
                                frameBitmaps.push(bitmap);
                            }}

                            if (!textureCanvas) {{
                                textureCanvas = document.createElement('canvas');
                                textureCanvas.width = captureCanvas.width;
                                textureCanvas.height = captureCanvas.height;
                                textureCtx = textureCanvas.getContext('2d');
                                texture = new THREE.CanvasTexture(textureCanvas);
                                texture.colorSpace = THREE.SRGBColorSpace;
                            }} else {{
                                textureCanvas.width = captureCanvas.width;
                                textureCanvas.height = captureCanvas.height;
                            }}

                            updateTexture(0);
                            updateStatus(`Loaded ${{videoName}}`);
                        }} catch (error) {{
                            console.error('Failed to load video frames', error);
                            updateStatus('フレーム抽出に失敗しました');
                        }} finally {{
                            loading = false;
                        }}
                    }}

                    async function cleanupFrames() {{
                        if (!frameBitmaps.length) return;
                        for (const bitmap of frameBitmaps) {{
                            if (bitmap.close) bitmap.close();
                        }}
                        frameBitmaps = [];
                    }}

                    function waitEvent(target, name) {{
                        return new Promise((resolve, reject) => {{
                            const onError = (err) => {{
                                target.removeEventListener('error', onError);
                                reject(err);
                            }};
                            target.addEventListener(name, function handler(event) {{
                                target.removeEventListener(name, handler);
                                target.removeEventListener('error', onError);
                                resolve(event);
                            }});
                            target.addEventListener('error', onError);
                        }});
                    }}

                    function seekVideo(video, time) {{
                        return new Promise((resolve) => {{
                            const onSeeked = () => {{
                                video.removeEventListener('seeked', onSeeked);
                                resolve();
                            }};
                            video.addEventListener('seeked', onSeeked);
                            video.currentTime = time;
                        }});
                    }}

                    function initThree() {{
                        if (renderer) return;
                        renderer = new THREE.WebGLRenderer({{ antialias: true }});
                        renderer.setPixelRatio(window.devicePixelRatio);
                        renderer.setSize(viewerEl.clientWidth, viewerEl.clientHeight);
                        viewerEl.appendChild(renderer.domElement);

                        scene = new THREE.Scene();
                        scene.background = new THREE.Color('#ffffff');

                        camera = new THREE.PerspectiveCamera(45, viewerEl.clientWidth / viewerEl.clientHeight, 0.1, 100);
                        camera.position.set(0, 0, 3);

                        controls = new OrbitControls(camera, renderer.domElement);
                        controls.enableDamping = true;
                        controls.enablePan = false;
                        controls.minDistance = 1.2;
                        controls.maxDistance = 5;
                        controls.minPolarAngle = THREE.MathUtils.degToRad(0.0001);
                        controls.maxPolarAngle = THREE.MathUtils.degToRad(90);
                        controls.addEventListener('change', handleCameraMove);

                        const light = new THREE.DirectionalLight(0xffffff, 1);
                        light.position.set(5, 5, 5);
                        scene.add(light);

                        const geometry = new THREE.BoxGeometry(1, 1.4, 0.2);
                        const material = new THREE.MeshBasicMaterial({{ map: texture }});
                        mesh = new THREE.Mesh(geometry, material);
                        scene.add(mesh);

                        window.addEventListener('resize', () => onResize());
                    }}

                    function updateTexture(index) {{
                        if (!textureCtx || !frameBitmaps.length) return;
                        const frame = frameBitmaps[Math.min(index, frameBitmaps.length - 1)];
                        textureCtx.drawImage(frame, 0, 0, textureCanvas.width, textureCanvas.height);
                        texture.needsUpdate = true;
                    }}

                    function handleCameraMove() {{
                        if (!frameBitmaps.length) return;
                        const {{ yaw, pitch }} = extractAngles();
                        const frameIndex = mapAnglesToFrame(yaw, pitch);
                        updateTexture(frameIndex);
                        anglesEl.textContent = `Yaw ${{yaw.toFixed(1)}}° / Pitch ${{pitch.toFixed(1)}}° / Frame ${{frameIndex}}`;
                    }}

                    function extractAngles() {{
                        const offset = new THREE.Vector3().copy(camera.position).sub(mesh.position);
                        const spherical = new THREE.Spherical().setFromVector3(offset);
                        const yaw = (THREE.MathUtils.radToDeg(spherical.theta) + 360) % 360;
                        const pitch = THREE.MathUtils.clamp(90 - THREE.MathUtils.radToDeg(spherical.phi), 0, 90);
                        return {{ yaw, pitch }};
                    }}

                    function mapAnglesToFrame(yaw, pitch) {{
                        const yawStep = 360 / 16;
                        const yawIndex = Math.floor(((yaw % 360) + yawStep / 2) / yawStep) % 16;
                        const band = PITCH_BANDS.find((segment) => pitch >= segment.min && pitch < segment.max) ?? PITCH_BANDS[PITCH_BANDS.length - 1];
                        let idx = band.offset + yawIndex;
                        return Math.min(idx, FRAME_COUNT - 1);
                    }}

                    function onResize() {{
                        if (!renderer || !camera) return;
                        const width = viewerEl.clientWidth;
                        const height = viewerEl.clientHeight;
                        renderer.setSize(width, height);
                        camera.aspect = width / height;
                        camera.updateProjectionMatrix();
                    }}

                    function animate() {{
                        requestAnimationFrame(animate);
                        if (controls) controls.update();
                        if (renderer && scene && camera) {{
                            renderer.render(scene, camera);
                        }}
                    }}

                    init();
                </script>
            </body>
        </html>
        """
    )

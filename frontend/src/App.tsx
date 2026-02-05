import { ChangeEvent, FormEvent, useCallback, useEffect, useRef, useState } from 'react';
import './App.css';
import { fetchStatus, fetchVideos, requestInference, type InferOptions, type VideoEntry } from './api';
import { ThreeViewer } from './components/ThreeViewer';

declare global {
  interface Window {
    gtag?: (...args: unknown[]) => void;
  }
}

function App() {
  const [videos, setVideos] = useState<VideoEntry[]>([]);
  const [extraVideos, setExtraVideos] = useState<VideoEntry[]>([]);
  const [selectedVideo, setSelectedVideo] = useState('');
  const [viewerAngles, setViewerAngles] = useState('');
  const [inferMessage, setInferMessage] = useState('');
  const [inferError, setInferError] = useState('');
  const [videoError, setVideoError] = useState('');
  const [engineReady, setEngineReady] = useState(true);
  const [engineError, setEngineError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [loadingVideos, setLoadingVideos] = useState(false);
  const [bgColor, setBgColor] = useState('#f8fafc');
  const [bgEnabled, setBgEnabled] = useState(true);
  const [inferSteps, setInferSteps] = useState(15);
  const [cfgScale, setCfgScale] = useState(1.0);
  const [loraMultiplier, setLoraMultiplier] = useState(1.5);
  const [prompt, setPrompt] = useState('360-degree orbit around the subject, camera rising in a spiral.');
  const [totalFrames] = useState(73);
  const [latentWindowSize] = useState(9);
  const [isDirty, setIsDirty] = useState(true);
  const [resultVideo, setResultVideo] = useState<VideoEntry | null>(null);
  const debugMode = (() => {
    const value = new URLSearchParams(window.location.search).get('debug');
    if (!value) return false;
    return value !== '0' && value.toLowerCase() !== 'false';
  })();

  const fileInputRef = useRef<HTMLInputElement>(null);

  const trackEvent = useCallback((name: string, params?: Record<string, unknown>) => {
    if (!window.gtag) return;
    window.gtag('event', name, params ?? {});
  }, []);

  const refreshStatus = useCallback(async () => {
    try {
      const status = await fetchStatus();
      setEngineReady(status.infer_ready);
      setEngineError(status.infer_error);
    } catch (error) {
      console.error('Failed to fetch status', error);
      setEngineError('推論エンジンの状態取得に失敗しました');
    }
  }, []);

  const mergeVideos = useCallback((base: VideoEntry[], extra: VideoEntry[]) => {
    const seen = new Set<string>();
    const merged: VideoEntry[] = [];
    for (const entry of [...base, ...extra]) {
      if (seen.has(entry.value)) continue;
      seen.add(entry.value);
      merged.push(entry);
    }
    return merged;
  }, []);

  const refreshVideos = useCallback(
    async (preferred?: string, extraOverride?: VideoEntry[]) => {
      setLoadingVideos(true);
      setVideoError('');
      try {
        const list = await fetchVideos();
        setVideos(list);
        const merged = mergeVideos(list, extraOverride ?? extraVideos);
        if (!merged.length) {
          setSelectedVideo('');
          return;
        }
        const resolved = preferred && merged.find((entry) => entry.value === preferred)
          ? preferred
          : merged[0].value;
        setSelectedVideo(resolved);
      } catch (error) {
        console.error('Failed to fetch videos', error);
        const message = error instanceof Error ? error.message : '動画リストの取得に失敗しました';
        setVideoError(message);
      } finally {
        setLoadingVideos(false);
      }
    },
    [extraVideos, mergeVideos],
  );

  useEffect(() => {
    refreshStatus();
    refreshVideos();
  }, [refreshStatus, refreshVideos]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!selectedVideo) {
      setViewerAngles('');
    }
  }, [selectedVideo]);

  const markDirty = () => {
    setIsDirty(true);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;
    setFile(nextFile);
    setInferMessage('');
    setInferError('');
    markDirty();
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : null);
  };

  const handleClear = () => {
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setInferMessage('');
    setInferError('');
    setIsDirty(true);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file || !engineReady || !isDirty) return;
    setIsUploading(true);
    setInferError('');
    try {
      trackEvent('generate_impostor_start');
      const options: InferOptions = {
        steps: inferSteps,
        cfg: cfgScale,
        loraMultiplier,
        prompt,
        totalFrames,
        latentWindowSize,
      };
      const result = await requestInference(file, options, debugMode);
      setInferMessage(result.message);
      setIsDirty(false);
      setResultVideo(result.video);
      trackEvent('generate_impostor_success', { steps: inferSteps, cfg: cfgScale, lora: loraMultiplier });
      const nextExtra = mergeVideos(extraVideos, [result.video]);
      setExtraVideos(nextExtra);
      setSelectedVideo(result.video.value);
      await refreshVideos(result.video.value, nextExtra);
      await refreshStatus();
    } catch (error) {
      console.error('Inference failed', error);
      const message = error instanceof Error ? error.message : '推論に失敗しました';
      setInferError(message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleAutoBgColor = useCallback((color: string) => {
    setBgColor(color);
    setBgEnabled(true);
  }, []);

  const triggerDownload = async (value: string) => {
    const isAbsolute = /^https?:\/\//i.test(value);
    const downloadUrl = isAbsolute ? value : `/${value}`;
    const filename = (value.split('/').pop() || 'impostor.mp4').split('?')[0] || 'impostor.mp4';
    const isSameOrigin = !isAbsolute || downloadUrl.startsWith(window.location.origin);

    if (isSameOrigin) {
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      link.rel = 'noopener';
      document.body.appendChild(link);
      link.click();
      link.remove();
      return;
    }

    try {
      const res = await fetch(downloadUrl, { mode: 'cors' });
      if (!res.ok) {
        throw new Error(`download failed: ${res.status}`);
      }
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = objectUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(objectUrl);
    } catch (error) {
      console.error('Auto download failed', error);
      window.open(downloadUrl, '_blank', 'noopener');
    }
  };

  const mergedVideos = mergeVideos(videos, extraVideos);

  return (
    <div className="app-shell">
      <header className="app-head">
        <h1>Impostor Maker</h1>
        <p className="lead">単一の画像から軽量な偽の3Dモデルを生成します。裏側で動画生成モデルを使用。</p>
      </header>
      <main className="layout">
        <section className="panel">
          <div className="panel-head">
            <h2>入力画像をアップロード</h2>
            <p className="muted">768x768 にリサイズして推論します。</p>
          </div>
          <form className="infer-form" onSubmit={handleSubmit}>
            <input
              ref={fileInputRef}
              className="file-input"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              disabled={!engineReady || isUploading}
              required
            />
            <div className="upload-preview">
              <div className="upload-preview-head">
                <button type="button" onClick={handleClear} disabled={!file}>
                  クリア
                </button>
              </div>
              <div className="preview-stage">
                {previewUrl ? (
                  <img src={previewUrl} alt="選択した画像のプレビュー" />
                ) : (
                  <p className="muted" style={{ textAlign: 'center' }}>
                    まだ画像が選択されていません。
                  </p>
                )}
              </div>
            </div>
            <div className="control-grid">
              <label className="control-field">
                Steps
                <input
                  type="number"
                  min={1}
                  max={200}
                  step={1}
                  value={inferSteps}
                  onChange={(event) => {
                    setInferSteps(Number(event.target.value));
                    markDirty();
                  }}
                  disabled={!engineReady || isUploading}
                />
              </label>
              <label className="control-field">
                CFG
                <input
                  type="number"
                  min={0}
                  max={20}
                  step={0.1}
                  value={cfgScale}
                  onChange={(event) => {
                    setCfgScale(Number(event.target.value));
                    markDirty();
                  }}
                  disabled={!engineReady || isUploading}
                />
              </label>
              <label className="control-field">
                LoRA multiplier
                <input
                  type="number"
                  min={0}
                  max={10}
                  step={0.05}
                  value={loraMultiplier}
                  onChange={(event) => {
                    setLoraMultiplier(Number(event.target.value));
                    markDirty();
                  }}
                  disabled={!engineReady || isUploading}
                />
              </label>
            </div>
            <details className="advanced-settings">
              <summary>詳細設定</summary>
              <div className="advanced-grid">
                <label className="control-field">
                  Prompt
                  <textarea
                    rows={3}
                    value={prompt}
                    onChange={(event) => {
                      setPrompt(event.target.value);
                      markDirty();
                    }}
                    disabled={!engineReady || isUploading}
                  />
                </label>
                <div className="advanced-row">
                  <label className="control-field">
                    Total Frames
                    <input type="number" value={totalFrames} disabled />
                  </label>
                  <label className="control-field">
                    Latent Window Size
                    <input type="number" value={latentWindowSize} disabled />
                  </label>
                </div>
              </div>
            </details>
            <button className="cta" type="submit" disabled={!file || !engineReady || isUploading || !isDirty}>
              {isUploading ? '推論中...' : isDirty ? '推論開始（約3分）' : '推論完了'}
            </button>
            <p className="muted note-text" style={{ textAlign: 'center' }}>
              サーバーに保存された画像・動画は、30日間で自動的に削除されます。
            </p>
          </form>
          {inferError && <p className="error-text">{inferError}</p>}
          {!engineReady && engineError && <p className="error-text">{engineError}</p>}
        </section>
        <section className="panel">
          <div className="panel-head">
            <h2>Impostor ビューア</h2>
            <p className="muted">カメラをドラッグすると角度に応じてフレームを切り替え、plane に貼り付けます。</p>
          </div>
          <label className="select-field">
            動画を選択
            <div className="select-row">
              <select
                value={selectedVideo}
                onChange={(event) => setSelectedVideo(event.target.value)}
                disabled={!mergedVideos.length || loadingVideos}
              >
                {mergedVideos.map((video) => (
                  <option key={video.value} value={video.value}>
                    {video.label}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="download-button"
                disabled={!selectedVideo}
                onClick={() => triggerDownload(selectedVideo)}
              >
                ダウンロード
              </button>
            </div>
          </label>
          <div className="color-row">
            <div className="color-row-head">
              <label className="color-label" htmlFor="chroma-key-color">
                クロマキー
              </label>
            </div>
            <div className="color-inputs">
              <input
                id="chroma-key-color"
                className="color-picker"
                type="color"
                value={bgColor}
                onChange={(event) => {
                  setBgColor(event.target.value);
                  setBgEnabled(true);
                }}
                disabled={!engineReady || isUploading}
              />
              <input
                className={`color-code${bgEnabled ? '' : ' is-muted'}`}
                type="text"
                value={bgEnabled ? bgColor.toUpperCase() : '未選択'}
                readOnly
              />
              {bgEnabled && (
                <button
                  className="ghost-button"
                  type="button"
                  onClick={() => setBgEnabled(false)}
                  aria-label="背景色選択をクリア"
                >
                  クリア
                </button>
              )}
            </div>
          </div>
          {loadingVideos && <p className="muted" style={{ fontSize: '0.85rem' }}>動画リストを更新中...</p>}
          {videoError && <p className="error-text">{videoError}</p>}
          <div className="viewer-shell">
            <ThreeViewer
              videoPath={selectedVideo || undefined}
              chromaKey={bgEnabled ? { color: bgColor, threshold: 0.08, softness: 0.08 } : undefined}
              showFloor
              onAnglesChange={setViewerAngles}
              onAutoKeyColor={handleAutoBgColor}
            />
          </div>
          <p className="status-text">{viewerAngles}</p>
        </section>
      </main>
      <footer className="app-footer">
        <div className="footer-content">
          <span>Made by</span>
          <a
            href="https://sawara.dev"
            target="_blank"
            rel="noopener noreferrer"
            className="footer-link"
          >
            <video
              ref={(video) => {
                if (!video) return;
                const playOnLoad = () => {
                  video.currentTime = 0;
                  video.play();
                };
                setTimeout(playOnLoad, 1000);
                video.onended = () => {
                  video.pause();
                };
              }}
              className="footer-avatar"
              muted
              playsInline
            >
              <source src="/images/logo.mp4" type="video/mp4" />
            </video>
          </a>
          <span>on</span>
          <a
            href="https://github.com/xhiroga/impostor"
            target="_blank"
            rel="noopener noreferrer"
            className="footer-link"
          >
            <img src="/images/github-mark.svg" alt="GitHub" className="footer-github" />
          </a>
          <span className="footer-sep">/</span>
          <a
            href="https://github.com/sponsors/xhiroga"
            target="_blank"
            rel="noopener noreferrer"
            className="footer-link is-sponsor"
            onClick={() => trackEvent('funding_link_click', { location: 'footer' })}
          >
            スポンサーを募集しています
            <svg
              className="footer-heart"
              viewBox="0 0 16 16"
              aria-hidden="true"
              focusable="false"
            >
              <path
                fill="currentColor"
                d="m8 14.25.345.666a.75.75 0 0 1-.69 0l-.008-.004-.018-.01a7.152 7.152 0 0 1-.31-.17 22.055 22.055 0 0 1-3.434-2.414C2.045 10.731 0 8.35 0 5.5 0 2.836 2.086 1 4.25 1 5.797 1 7.153 1.802 8 3.02 8.847 1.802 10.203 1 11.75 1 13.914 1 16 2.836 16 5.5c0 2.85-2.045 5.231-3.885 6.818a22.066 22.066 0 0 1-3.744 2.584l-.018.01-.006.003h-.002ZM4.25 2.5c-1.336 0-2.75 1.164-2.75 3 0 2.15 1.58 4.144 3.365 5.682A20.58 20.58 0 0 0 8 13.393a20.58 20.58 0 0 0 3.135-2.211C12.92 9.644 14.5 7.65 14.5 5.5c0-1.836-1.414-3-2.75-3-1.373 0-2.609.986-3.029 2.456a.749.749 0 0 1-1.442 0C6.859 3.486 5.623 2.5 4.25 2.5Z"
              />
            </svg>
          </a>
        </div>
        <p className="footer-note">アクセス解析にGoogle Analyticsを使用しています。</p>
      </footer>
      {resultVideo && (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <div className="modal-card">
            <h3>推論が完了しました</h3>
            <p className="muted">生成した動画はビューアで確認できます。</p>
            <div className="modal-actions">
              <button
                type="button"
                className="cta"
                onClick={() => {
                  trackEvent('download_click', { location: 'result_modal' });
                  triggerDownload(resultVideo.value);
                }}
              >
                生成した動画をダウンロード
              </button>
              <a
                className="secondary-button sponsor-button"
                href="https://github.com/sponsors/xhiroga"
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => trackEvent('funding_link_click', { location: 'result_modal' })}
              >
                この研究を支援する
              </a>
            </div>
            <button
              type="button"
              className="ghost-button"
              onClick={() => setResultVideo(null)}
            >
              閉じる
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

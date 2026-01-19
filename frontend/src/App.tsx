import { ChangeEvent, FormEvent, useCallback, useEffect, useRef, useState } from 'react';
import './App.css';
import { fetchStatus, fetchVideos, requestInference, type VideoEntry } from './api';
import { ThreeViewer } from './components/ThreeViewer';

function App() {
  const [videos, setVideos] = useState<VideoEntry[]>([]);
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

  const fileInputRef = useRef<HTMLInputElement>(null);

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

  const refreshVideos = useCallback(
    async (preferred?: string) => {
      setLoadingVideos(true);
      setVideoError('');
      try {
        const list = await fetchVideos();
        setVideos(list);
        if (!list.length) {
          setSelectedVideo('');
          return;
        }
        const resolved = preferred && list.find((entry) => entry.value === preferred)
          ? preferred
          : list[0].value;
        setSelectedVideo(resolved);
      } catch (error) {
        console.error('Failed to fetch videos', error);
        const message = error instanceof Error ? error.message : '動画リストの取得に失敗しました';
        setVideoError(message);
      } finally {
        setLoadingVideos(false);
      }
    },
    [],
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

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;
    setFile(nextFile);
    setInferMessage('');
    setInferError('');
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
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file || !engineReady) return;
    setIsUploading(true);
    setInferError('');
    try {
      const result = await requestInference(file);
      setInferMessage(result.message);
      await refreshVideos(result.video.value);
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

  return (
    <div className="app-shell">
      <header className="app-head">
        <h1>Impostor Maker</h1>
        <p className="lead">単一の画像から軽量な偽の3Dモデルを生成します。裏側で動画生成モデルを使用。</p>
      </header>
      <main className="layout">
        <section className="panel">
          <div className="panel-head">
            <p className="eyebrow">Upload</p>
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
                <span className="muted" style={{ fontWeight: 600, color: 'var(--fg)' }}>
                  入力画像プレビュー
                </span>
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
            <button className="cta" type="submit" disabled={!file || !engineReady || isUploading}>
              {isUploading ? '推論中...' : '推論スタート'}
            </button>
          </form>
          {inferMessage && <p className="success-text">{inferMessage}</p>}
          {inferError && <p className="error-text">{inferError}</p>}
          {!engineReady && engineError && <p className="error-text">{engineError}</p>}
        </section>
        <section className="panel">
          <div className="panel-head">
            <p className="eyebrow">Preview</p>
            <h2>Impostor ビューア</h2>
            <p className="muted">カメラをドラッグすると角度に応じてフレームを切り替え、plane に貼り付けます。</p>
          </div>
          <label className="select-field">
            動画セレクタ
            <select
              value={selectedVideo}
              onChange={(event) => setSelectedVideo(event.target.value)}
              disabled={!videos.length || loadingVideos}
            >
              {videos.map((video) => (
                <option key={video.value} value={video.value}>
                  {video.label}
                </option>
              ))}
            </select>
          </label>
          <div className="color-row">
            <label className="color-field">
              背景色
              <div className="color-inputs">
                <input
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
              </div>
            </label>
            <div className="color-actions">
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
              chromaKey={bgEnabled ? { color: bgColor, threshold: 0.12, softness: 0.1 } : undefined}
              showFloor
              onAnglesChange={setViewerAngles}
              onAutoKeyColor={handleAutoBgColor}
            />
          </div>
          <p className="status-text">{viewerAngles}</p>
        </section>
      </main>
    </div>
  );
}

export default App;

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const FRAME_COUNT = 81;
const PITCH_BANDS = [
  { min: 0, max: 18, offset: 0 },
  { min: 18, max: 36, offset: 16 },
  { min: 36, max: 54, offset: 32 },
  { min: 54, max: 72, offset: 48 },
  { min: 72, max: 91, offset: 64 },
];
const PANEL_BASE_SIZE = 1.2;

export type ChromaKeySettings = {
  color: string;
  threshold?: number;
  softness?: number;
};

export type ThreeViewerProps = {
  videoPath?: string;
  onAnglesChange?: (text: string) => void;
  chromaKey?: ChromaKeySettings;
  transparentBackground?: boolean;
  showFloor?: boolean;
  onAutoKeyColor?: (color: string) => void;
};

type ViewerState = {
  renderer?: THREE.WebGLRenderer;
  scene?: THREE.Scene;
  camera?: THREE.PerspectiveCamera;
  controls?: OrbitControls;
  panelMesh?: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  floorMesh?: THREE.Mesh;
  texture?: THREE.CanvasTexture;
  textureCanvas?: HTMLCanvasElement;
  textureCtx?: CanvasRenderingContext2D | null;
  frameBitmaps: ImageBitmap[];
  panelDimensions: { width: number; height: number };
  keyEnabled?: boolean;
  autoSampledFor?: string;
};

const parseHexColor = (value: string) => {
  const hex = value.replace('#', '').trim();
  if (hex.length !== 6) {
    return { r: 255, g: 255, b: 255 };
  }
  const num = Number.parseInt(hex, 16);
  return {
    r: (num >> 16) & 255,
    g: (num >> 8) & 255,
    b: num & 255,
  };
};

const clampByte = (value: number) => Math.max(0, Math.min(255, Math.round(value)));

const toHexColor = (r: number, g: number, b: number) => {
  const toHex = (value: number) => clampByte(value).toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
};

const sampleRegionColor = (ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number) => {
  const data = ctx.getImageData(x, y, width, height).data;
  let r = 0;
  let g = 0;
  let b = 0;
  let count = 0;
  for (let i = 0; i < data.length; i += 4) {
    r += data[i];
    g += data[i + 1];
    b += data[i + 2];
    count += 1;
  }
  if (!count) return { r: 255, g: 255, b: 255 };
  return { r: r / count, g: g / count, b: b / count };
};

const inferBackgroundColor = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
  const sampleSize = Math.max(4, Math.floor(Math.min(width, height) * 0.08));
  const samples = [
    sampleRegionColor(ctx, 0, 0, sampleSize, sampleSize),
    sampleRegionColor(ctx, width - sampleSize, 0, sampleSize, sampleSize),
    sampleRegionColor(ctx, 0, height - sampleSize, sampleSize, sampleSize),
    sampleRegionColor(ctx, width - sampleSize, height - sampleSize, sampleSize, sampleSize),
  ];
  const avg = samples.reduce(
    (acc, sample) => ({
      r: acc.r + sample.r / samples.length,
      g: acc.g + sample.g / samples.length,
      b: acc.b + sample.b / samples.length,
    }),
    { r: 0, g: 0, b: 0 },
  );
  return toHexColor(avg.r, avg.g, avg.b);
};

const applyChromaKey = (
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  key: ChromaKeySettings,
) => {
  const { r: keyR, g: keyG, b: keyB } = parseHexColor(key.color);
  const threshold = Math.max(0, Math.min(1, key.threshold ?? 0.12));
  const softness = Math.max(0, Math.min(1, key.softness ?? 0.08));
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const maxDistance = Math.sqrt(3 * 255 * 255);
  for (let i = 0; i < data.length; i += 4) {
    const dr = data[i] - keyR;
    const dg = data[i + 1] - keyG;
    const db = data[i + 2] - keyB;
    const distance = Math.sqrt(dr * dr + dg * dg + db * db) / maxDistance;
    let alpha = 1;
    if (distance <= threshold) {
      alpha = 0;
    } else if (softness > 0) {
      alpha = Math.min(1, (distance - threshold) / softness);
    }
    data[i + 3] = Math.round(data[i + 3] * alpha);
  }
  ctx.putImageData(imageData, 0, 0);
};

export function ThreeViewer({
  videoPath,
  onAnglesChange,
  chromaKey,
  transparentBackground,
  showFloor = true,
  onAutoKeyColor,
}: ThreeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<ViewerState>({
    frameBitmaps: [],
    panelDimensions: { width: PANEL_BASE_SIZE, height: PANEL_BASE_SIZE },
  });
  const [ready, setReady] = useState(false);

  const updateAngles = (text: string) => {
    if (onAnglesChange) {
      onAnglesChange(text);
    }
  };

  const cleanupFrames = () => {
    const state = stateRef.current;
    if (state.frameBitmaps.length) {
      for (const bitmap of state.frameBitmaps) {
        if (bitmap.close) {
          bitmap.close();
        }
      }
    }
    state.frameBitmaps = [];
  };

  const updateTexture = (index: number) => {
    const state = stateRef.current;
    if (!state.textureCtx || !state.textureCanvas || !state.texture || !state.frameBitmaps.length) {
      if (!state.frameBitmaps.length) {
        console.warn('[viewer] updateTexture skipped (no frames)');
      }
      return;
    }
    const frame = state.frameBitmaps[Math.min(index, state.frameBitmaps.length - 1)];
    state.textureCtx.clearRect(0, 0, state.textureCanvas.width, state.textureCanvas.height);
    state.textureCtx.drawImage(frame, 0, 0, state.textureCanvas.width, state.textureCanvas.height);
    state.texture.needsUpdate = true;
  };

  const extractAngles = () => {
    const state = stateRef.current;
    if (!state.camera || !state.panelMesh) {
      return { yaw: 0, pitch: 0 };
    }
    const offset = new THREE.Vector3().copy(state.camera.position).sub(state.panelMesh.position);
    const spherical = new THREE.Spherical().setFromVector3(offset);
    const yaw = (THREE.MathUtils.radToDeg(spherical.theta) + 360) % 360;
    const pitch = THREE.MathUtils.clamp(90 - THREE.MathUtils.radToDeg(spherical.phi), 0, 90);
    return { yaw, pitch };
  };

  const mapAnglesToFrame = (yaw: number, pitch: number) => {
    const yawStep = 360 / 16;
    const yawReversed = (360 - yaw) % 360;
    const yawIndex = Math.floor(((yawReversed % 360) + yawStep / 2) / yawStep) % 16;
    const band =
      PITCH_BANDS.find((segment) => pitch >= segment.min && pitch < segment.max) ?? PITCH_BANDS[PITCH_BANDS.length - 1];
    return Math.min(band.offset + yawIndex, FRAME_COUNT - 1);
  };

  const handleCameraMove = () => {
    const state = stateRef.current;
    if (!state.frameBitmaps.length) {
      return;
    }
    const { yaw, pitch } = extractAngles();
    const frameIndex = mapAnglesToFrame(yaw, pitch);
    updateTexture(frameIndex);
    updateAngles(`Yaw ${yaw.toFixed(1)}° / Pitch ${pitch.toFixed(1)}° / Frame ${frameIndex}`);
  };

  const updatePanelDimensions = (textureWidth: number, textureHeight: number) => {
    if (!textureWidth || !textureHeight) return;
    const height = PANEL_BASE_SIZE;
    const width = PANEL_BASE_SIZE * (textureWidth / textureHeight);
    stateRef.current.panelDimensions = { width, height };
    rebuildPanelMesh();
  };

  const rebuildPanelMesh = () => {
    const state = stateRef.current;
    if (!state.scene || !state.texture) return;
    if (state.panelMesh) {
      state.scene.remove(state.panelMesh);
      state.panelMesh.geometry.dispose();
      state.panelMesh.material.dispose();
    }
    const geometry = new THREE.PlaneGeometry(state.panelDimensions.width, state.panelDimensions.height);
    const material = new THREE.MeshBasicMaterial({
      map: state.texture,
      side: THREE.DoubleSide,
      transparent: Boolean(state.keyEnabled),
      alphaTest: state.keyEnabled ? 0.02 : 0,
    });
    state.panelMesh = new THREE.Mesh(geometry, material);
    state.panelMesh.position.y = -0.4;
    state.scene.add(state.panelMesh);
  };

  const applyFloorTextures = () => {
    const state = stateRef.current;
    if (!state.floorMesh) return;
    const loader = new THREE.TextureLoader();
    loader.load(
      '/assets/green-grass.png',
      (ground) => {
        ground.wrapS = THREE.MirroredRepeatWrapping;
        ground.wrapT = THREE.MirroredRepeatWrapping;
        ground.repeat.set(1.5, 1.5);
        ground.colorSpace = THREE.SRGBColorSpace;
        const material = state.floorMesh!.material as THREE.MeshStandardMaterial;
        material.map = ground;
        material.roughness = 0.9;
        material.needsUpdate = true;
      },
      undefined,
      (error) => {
        console.warn('Failed to load floor texture', error);
      },
    );
  };

  const handleResize = () => {
    const state = stateRef.current;
    const container = containerRef.current;
    if (!state.renderer || !state.camera || !container) return;
    const width = container.clientWidth;
    const height = container.clientHeight || width * (9 / 16);
    state.renderer.setSize(width, height);
    state.camera.aspect = width / height;
    state.camera.updateProjectionMatrix();
  };

  const waitEvent = (video: HTMLVideoElement, name: string, isCancelled: () => boolean) => {
    return new Promise<void>((resolve, reject) => {
      const cleanup = () => {
        video.removeEventListener(name, onEvent);
        video.removeEventListener('error', onError);
      };

      const onError = (event: Event) => {
        cleanup();
        reject(event instanceof Error ? event : new Error('video error'));
      };

      const onEvent = () => {
        cleanup();
        if (isCancelled()) {
          reject(new Error('cancelled'));
        } else {
          resolve();
        }
      };

      video.addEventListener(name, onEvent, { once: true });
      video.addEventListener('error', onError, { once: true });
    });
  };

  const seekVideo = (video: HTMLVideoElement, time: number, isCancelled: () => boolean) => {
    return new Promise<void>((resolve, reject) => {
      const onSeeked = () => {
        video.removeEventListener('seeked', onSeeked);
        if (isCancelled()) {
          reject(new Error('cancelled'));
        } else {
          resolve();
        }
      };
      video.addEventListener('seeked', onSeeked, { once: true });
      video.currentTime = time;
    });
  };

  const waitForCurrentData = (video: HTMLVideoElement, isCancelled: () => boolean) => {
    if (video.readyState >= video.HAVE_CURRENT_DATA) {
      return Promise.resolve();
    }
    return new Promise<void>((resolve, reject) => {
      const cleanup = () => {
        video.removeEventListener('canplay', onCanPlay);
        video.removeEventListener('error', onError);
      };
      const onCanPlay = () => {
        cleanup();
        if (isCancelled()) {
          reject(new Error('cancelled'));
        } else {
          resolve();
        }
      };
      const onError = (err: Event) => {
        cleanup();
        reject(err instanceof Error ? err : new Error('video error'));
      };
      video.addEventListener('canplay', onCanPlay, { once: true });
      video.addEventListener('error', onError, { once: true });
    });
  };

  const loadVideo = async (path: string, isCancelled: () => boolean) => {
    const state = stateRef.current;
    if (!state.textureCanvas || !state.textureCtx || !state.texture) return;
    if (typeof window === 'undefined' || typeof window.createImageBitmap !== 'function') {
      console.warn('[viewer] createImageBitmap is not supported in this environment');
      return;
    }

    console.log('[viewer] start loading video', path);
    cleanupFrames();

    const source = path.startsWith('http') ? path : `/${path.replace(/^\//, '')}`;
    const video = document.createElement('video');
    video.crossOrigin = 'anonymous';
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';
    video.src = source;
    video.load();

    await waitEvent(video, 'loadedmetadata', isCancelled);
    if (isCancelled()) return;

    if (!Number.isFinite(video.videoWidth) || !Number.isFinite(video.videoHeight) || video.videoWidth <= 0 || video.videoHeight <= 0) {
      throw new Error(`動画のメタデータ取得に失敗しました (width=${video.videoWidth}, height=${video.videoHeight})`);
    }

    const videoWidth = Math.max(video.videoWidth, 1);
    const videoHeight = Math.max(video.videoHeight, 1);
    console.log('[viewer] metadata', { videoWidth, videoHeight, duration: video.duration });
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = videoWidth;
    captureCanvas.height = videoHeight;
    const captureCtx = captureCanvas.getContext('2d');
    if (!captureCtx) {
      throw new Error('Canvas 2D context を初期化できません');
    }
    captureCtx.imageSmoothingEnabled = true;
    captureCtx.imageSmoothingQuality = 'high';

    const duration = video.duration || 0;
    const maxTime = duration > 0 ? duration - 0.0001 : 0;
    const frameTimes = Array.from({ length: FRAME_COUNT }, (_, i) => Math.min((i / (FRAME_COUNT - 1)) * duration, maxTime));
    const keySettings = chromaKey?.color ? chromaKey : undefined;
    state.keyEnabled = Boolean(keySettings);
    const shouldAutoSample = Boolean(onAutoKeyColor) && state.autoSampledFor !== path;
    let didAutoSample = false;

    for (const time of frameTimes) {
      await seekVideo(video, time, isCancelled);
      await waitForCurrentData(video, isCancelled);
      if (isCancelled()) return;
      captureCtx.clearRect(0, 0, videoWidth, videoHeight);
      captureCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, videoWidth, videoHeight);
      if (!didAutoSample && shouldAutoSample && onAutoKeyColor) {
        const inferred = inferBackgroundColor(captureCtx, videoWidth, videoHeight);
        onAutoKeyColor(inferred);
        didAutoSample = true;
        state.autoSampledFor = path;
      }
      if (keySettings) {
        applyChromaKey(captureCtx, videoWidth, videoHeight, keySettings);
      }
      let bitmap: ImageBitmap;
      try {
        bitmap = await window.createImageBitmap(captureCanvas);
      } catch (err) {
        console.error('[viewer] createImageBitmap failed', err);
        throw err;
      }
      if (isCancelled()) {
        if (bitmap.close) bitmap.close();
        return;
      }
      state.frameBitmaps.push(bitmap);
    }

    console.log('[viewer] extracted frames', state.frameBitmaps.length);

    state.textureCanvas.width = captureCanvas.width;
    state.textureCanvas.height = captureCanvas.height;
    if (state.texture) {
      state.texture.dispose();
    }
    state.texture = new THREE.CanvasTexture(state.textureCanvas);
    state.texture.premultiplyAlpha = Boolean(keySettings);
    state.texture.colorSpace = THREE.SRGBColorSpace;
    rebuildPanelMesh();
    updatePanelDimensions(captureCanvas.width, captureCanvas.height);
    updateTexture(0);
    console.log('[viewer] texture updated', {
      texWidth: state.textureCanvas.width,
      texHeight: state.textureCanvas.height,
    });
  };

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const state = stateRef.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    const width = container.clientWidth || 960;
    const height = container.clientHeight || width * (9 / 16);
    renderer.setSize(width, height);
    if (transparentBackground) {
      renderer.setClearColor(0x000000, 0);
    }
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    if (!transparentBackground) {
      scene.background = new THREE.Color('#ffffff');
    }

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 1, 2.4);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 1.4;
    controls.maxDistance = 5;
    controls.minPolarAngle = THREE.MathUtils.degToRad(0.0001);
    controls.maxPolarAngle = THREE.MathUtils.degToRad(90);
    controls.addEventListener('change', handleCameraMove);

    const textureCanvas = document.createElement('canvas');
    textureCanvas.width = 1;
    textureCanvas.height = 1;
    const textureCtx = textureCanvas.getContext('2d');
    textureCtx?.fillRect(0, 0, 1, 1);
    const texture = new THREE.CanvasTexture(textureCanvas);
    texture.colorSpace = THREE.SRGBColorSpace;

    state.renderer = renderer;
    state.scene = scene;
    state.camera = camera;
    state.controls = controls;
    state.texture = texture;
    state.textureCanvas = textureCanvas;
    state.textureCtx = textureCtx;

    const light = new THREE.DirectionalLight(0xfff3d4, 1.2);
    light.position.set(5, 6, 4);
    scene.add(light);
    const ambient = new THREE.HemisphereLight(0xf9f5df, 0x5b4d2b, 0.6);
    scene.add(ambient);

    rebuildPanelMesh();

    let floorGeometry: THREE.PlaneGeometry | null = null;
    let floorMaterial: THREE.MeshStandardMaterial | null = null;
    if (showFloor) {
      floorGeometry = new THREE.PlaneGeometry(6, 6);
      floorMaterial = new THREE.MeshStandardMaterial({ color: '#ffffff', metalness: 0.05, roughness: 0.9 });
      const floorMesh = new THREE.Mesh(floorGeometry, floorMaterial);
      floorMesh.rotation.x = -Math.PI / 2;
      floorMesh.position.y = -1;
      scene.add(floorMesh);
      state.floorMesh = floorMesh;
      applyFloorTextures();
    }

    const onResize = () => handleResize();
    window.addEventListener('resize', onResize);

    let animationFrame = 0;
    const animate = () => {
      animationFrame = requestAnimationFrame(animate);
      controls.update();
      if (state.panelMesh && state.camera) {
        state.panelMesh.lookAt(state.camera.position);
      }
      renderer.render(scene, camera);
    };
    animate();

    setReady(true);
    return () => {
      setReady(false);
      window.removeEventListener('resize', onResize);
      cancelAnimationFrame(animationFrame);
      controls.removeEventListener('change', handleCameraMove);
      cleanupFrames();
      renderer.dispose();
      texture.dispose();
      floorGeometry?.dispose();
      floorMaterial?.dispose();
      if (state.panelMesh) {
        state.panelMesh.geometry.dispose();
        state.panelMesh.material.dispose();
      }
      state.renderer = undefined;
      state.scene = undefined;
      state.camera = undefined;
      state.controls = undefined;
      state.panelMesh = undefined;
      state.floorMesh = undefined;
      state.texture = undefined;
      state.textureCanvas = undefined;
      state.textureCtx = undefined;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!ready) return;
    if (!videoPath) {
      cleanupFrames();
      updateAngles('');
      return;
    }

    let cancelled = false;
    const abortFlag = () => cancelled;
    (async () => {
      try {
        await loadVideo(videoPath, abortFlag);
      } catch (error) {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : String(error);
        console.error('[viewer] failed to load video', message);
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    videoPath,
    ready,
    chromaKey?.color,
    chromaKey?.threshold,
    chromaKey?.softness,
    onAutoKeyColor,
  ]);

  return <div className="three-viewer" ref={containerRef} />;
}

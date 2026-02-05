export type VideoEntry = {
  value: string;
  label: string;
};

export type StatusResponse = {
  infer_ready: boolean;
  infer_error: string | null;
};

export type InferResponse = {
  message: string;
  video: VideoEntry;
  upload_filename: string;
};

export type InferOptions = {
  steps?: number;
  cfg?: number;
  loraMultiplier?: number;
  prompt?: string;
  totalFrames?: number;
  latentWindowSize?: number;
};

async function handleJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `API request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchStatus(): Promise<StatusResponse> {
  const res = await fetch('/api/status');
  return handleJson<StatusResponse>(res);
}

export async function fetchVideos(): Promise<VideoEntry[]> {
  const res = await fetch('/api/videos');
  return handleJson<VideoEntry[]>(res);
}

export async function requestInference(
  file: File,
  options: InferOptions,
  debug?: boolean,
): Promise<InferResponse> {
  const form = new FormData();
  form.append('image', file, file.name);
  if (typeof options.steps === 'number') {
    form.append('steps', options.steps.toString());
  }
  if (typeof options.cfg === 'number') {
    form.append('cfg', options.cfg.toString());
  }
  if (typeof options.loraMultiplier === 'number') {
    form.append('lora_multiplier', options.loraMultiplier.toString());
  }
  if (options.prompt) {
    form.append('prompt', options.prompt);
  }
  if (typeof options.totalFrames === 'number') {
    form.append('total_frames', options.totalFrames.toString());
  }
  if (typeof options.latentWindowSize === 'number') {
    form.append('latent_window_size', options.latentWindowSize.toString());
  }
  const query = debug ? '?debug=1' : '';
  const res = await fetch(`/api/infer${query}`, {
    method: 'POST',
    body: form,
  });
  return handleJson<InferResponse>(res);
}

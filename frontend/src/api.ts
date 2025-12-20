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

export async function requestInference(file: File): Promise<InferResponse> {
  const form = new FormData();
  form.append('image', file, file.name);
  const res = await fetch('/api/infer', {
    method: 'POST',
    body: form,
  });
  return handleJson<InferResponse>(res);
}

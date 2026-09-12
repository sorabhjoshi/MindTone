import { useCallback, useRef, useState } from 'react';

/**
 * Wraps the browser's MediaRecorder API. Returns recording state, a
 * playable object URL once stopped, the raw Blob (for upload), and
 * start/stop/reset controls.
 */
export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [error, setError] = useState(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const timerRef = useRef(null);

  const start = useCallback(async () => {
    setError(null);
    setAudioUrl(null);
    setAudioBlob(null);
    chunksRef.current = [];

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Your browser does not support microphone recording. Try uploading a file instead.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      // audio/webm is broadly supported across Chrome/Firefox/Edge; the
      // backend accepts it directly (soundfile/torchaudio both handle it
      // via their respective decoders, with a fallback in place).
      const mimeType = MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : '';
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach((t) => t.stop());
        clearInterval(timerRef.current);
      };

      recorder.start();
      setIsRecording(true);
      setElapsedSeconds(0);
      timerRef.current = setInterval(() => setElapsedSeconds((s) => s + 1), 1000);
    } catch {
      setError('Could not access your microphone. Check browser permissions, or upload a file instead.');
    }
  }, []);

  const stop = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  }, []);

  const reset = useCallback(() => {
    setAudioUrl(null);
    setAudioBlob(null);
    setError(null);
    setElapsedSeconds(0);
  }, []);

  return { isRecording, audioUrl, audioBlob, error, elapsedSeconds, start, stop, reset, setAudioBlob, setAudioUrl };
}

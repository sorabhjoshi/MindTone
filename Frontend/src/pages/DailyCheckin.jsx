import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import Layout from '../components/Layout';
import EmotionBar from '../components/EmotionBar';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import { api } from '../api/client';

export default function DailyCheckin() {
  const [prompt, setPrompt] = useState('');
  const [promptLoading, setPromptLoading] = useState(true);
  const [todayCount, setTodayCount] = useState(0);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const recorder = useAudioRecorder();

  const loadPrompt = useCallback(async (exclude) => {
    setPromptLoading(true);
    try {
      const sentence = await api.getPrompt(exclude);
      setPrompt(sentence);
    } finally {
      setPromptLoading(false);
    }
  }, []);

  const loadTodayCount = useCallback(async () => {
    try {
      const checkins = await api.getTodayCheckins();
      setTodayCount(checkins.length);
    } catch {
      // non-critical — just skip showing the count
    }
  }, []);

  useEffect(() => {
    loadPrompt();
    loadTodayCount();
  }, [loadPrompt, loadTodayCount]);

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (file) {
      recorder.reset();
      setUploadedFile(file);
    }
  }

  const activeBlob = recorder.audioBlob || uploadedFile;
  const activeUrl = recorder.audioUrl || (uploadedFile ? URL.createObjectURL(uploadedFile) : null);

  async function handleSubmit() {
    if (!activeBlob) return;
    setSubmitting(true);
    setError('');
    const { data, error } = await api.submitCheckin(prompt, activeBlob);
    setSubmitting(false);
    if (error) {
      setError(error);
      return;
    }
    setResult(data);
    setTodayCount((c) => c + 1);
    recorder.reset();
    setUploadedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    loadPrompt(prompt); // roll a new sentence, excluding the one just used
  }

  function handleNewPrompt() {
    loadPrompt(prompt);
  }

  return (
    <Layout>
      <h1 className="text-2xl font-bold text-teal-900">🎙️ Daily Mood Check-in</h1>
      <p className="mt-1 text-sm text-teal-900/60">
        This tracks your detected speech emotion only — long-term patterns show up on the{' '}
        <Link to="/history" className="font-medium text-teal-600 underline">History &amp; Trends</Link> page
        once you have enough check-ins.
      </p>

      {todayCount > 0 && (
        <div className="mt-4 rounded-xl bg-teal-50 px-4 py-2.5 text-sm text-teal-900/80">
          You&rsquo;ve checked in {todayCount} time{todayCount !== 1 ? 's' : ''} today. You can check in
          again any time — every entry is kept, none get overwritten.
        </div>
      )}

      {/* Prompt card */}
      <div className="animate-fade-in-up mt-6 rounded-2xl border border-teal-100 bg-white p-6 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-teal-500">
              Read this sentence out loud
            </p>
            <p className="mt-2 text-xl font-medium text-teal-900">
              {promptLoading ? 'Loading...' : `"${prompt}"`}
            </p>
          </div>
          <button
            onClick={handleNewPrompt}
            disabled={promptLoading}
            className="shrink-0 rounded-full border border-teal-200 px-3 py-1.5 text-xs font-medium text-teal-700 hover:bg-teal-50"
          >
            🔀 Different sentence
          </button>
        </div>
      </div>

      {/* Recording card */}
      <div className="animate-fade-in-up mt-4 rounded-2xl border border-teal-100 bg-white p-6 shadow-sm">
        <p className="mb-4 text-sm font-semibold text-teal-900">Record your check-in</p>

        {recorder.error && (
          <div className="mb-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{recorder.error}</div>
        )}

        <div className="flex flex-wrap items-center gap-3">
          {!recorder.isRecording ? (
            <button
              onClick={recorder.start}
              className="flex items-center gap-2 rounded-full bg-teal-500 px-5 py-2.5 text-sm font-semibold text-white hover:bg-teal-600"
            >
              🎙️ Start recording
            </button>
          ) : (
            <button
              onClick={recorder.stop}
              className="flex items-center gap-2 rounded-full bg-red-500 px-5 py-2.5 text-sm font-semibold text-white animate-pulse-ring"
            >
              ⏹ Stop ({recorder.elapsedSeconds}s)
            </button>
          )}

          <span className="text-sm text-teal-900/40">or</span>

          <label className="cursor-pointer rounded-full border border-teal-200 px-4 py-2 text-sm font-medium text-teal-700 hover:bg-teal-50">
            📁 Upload a file
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>
        </div>

        {activeUrl && (
          <div className="mt-4">
            <audio controls src={activeUrl} className="w-full" />
            <button
              onClick={handleSubmit}
              disabled={submitting}
              className="mt-3 w-full rounded-lg bg-teal-500 py-2.5 text-sm font-semibold text-white hover:bg-teal-600 disabled:opacity-60"
            >
              {submitting ? 'Analyzing...' : 'Analyze and save'}
            </button>
          </div>
        )}

        {error && <div className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>}
      </div>

      {/* Result card */}
      {result && (
        <div className="animate-fade-in-up mt-4 rounded-2xl border border-teal-100 bg-white p-6 shadow-sm">
          <p className="mb-4 text-sm font-semibold text-teal-900">✅ Saved! Result</p>
          <div className="mb-4 grid grid-cols-2 gap-4">
            <div className="rounded-xl bg-teal-50 p-4 text-center">
              <p className="text-xs text-teal-900/50">Detected emotion</p>
              <p className="mt-1 text-lg font-bold text-teal-900">{result.predicted_emotion}</p>
              <p className="text-xs text-teal-900/50">{(result.confidence * 100).toFixed(0)}% confidence</p>
            </div>
            <div className="rounded-xl bg-teal-50 p-4 text-center">
              <p className="text-xs text-teal-900/50">Uncertainty</p>
              <p className="mt-1 text-lg font-bold text-teal-900">{result.uncertainty.toFixed(2)}</p>
            </div>
          </div>

          <p className="mb-2 text-sm font-semibold text-teal-900">Full probability breakdown</p>
          {['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad'].map((emotion) => (
            <EmotionBar
              key={emotion}
              emotion={emotion}
              probability={
                {
                  Angry: result.prob_angry,
                  Disgust: result.prob_disgust,
                  Fear: result.prob_fear,
                  Happy: result.prob_happy,
                  Neutral: result.prob_neutral,
                  Sad: result.prob_sad,
                }[emotion]
              }
            />
          ))}

          <Link
            to="/history"
            className="mt-3 inline-block text-sm font-medium text-teal-600 underline"
          >
            See your long-term mood patterns →
          </Link>
        </div>
      )}
    </Layout>
  );
}

import { useCallback, useEffect, useState } from 'react';
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip,
  XAxis, YAxis, Bar, BarChart, Cell,
} from 'recharts';
import Layout from '../components/Layout';
import { EMOTION_COLORS } from '../components/EmotionBar';
import { api } from '../api/client';

const RANGE_OPTIONS = [
  { label: 'Last 30 days', days: 30 },
  { label: 'Last 90 days', days: 90 },
  { label: 'Last 365 days', days: 365 },
  { label: 'All time', days: null },
];

export default function History() {
  const [rangeIdx, setRangeIdx] = useState(1);
  const [checkins, setCheckins] = useState([]);
  const [pattern, setPattern] = useState(null);
  const [loading, setLoading] = useState(true);

  const days = RANGE_OPTIONS[rangeIdx].days;

  const load = useCallback(async () => {
    setLoading(true);
    const [history, patternData] = await Promise.all([
      api.getHistory(days),
      api.getPatterns(days),
    ]);
    setCheckins(history);
    setPattern(patternData);
    setLoading(false);
  }, [days]);

  useEffect(() => {
    load();
  }, [load]);

  const emotionCols = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad'];

  const lineData = checkins.map((c) => ({
    time: new Date(c.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
    Angry: c.prob_angry, Disgust: c.prob_disgust, Fear: c.prob_fear,
    Happy: c.prob_happy, Neutral: c.prob_neutral, Sad: c.prob_sad,
  }));

  const distribution = emotionCols
    .map((e) => ({ emotion: e, count: checkins.filter((c) => c.predicted_emotion === e).length }))
    .filter((d) => d.count > 0);

  const streak = computeStreak(checkins);
  const mostCommon = distribution.length
    ? distribution.reduce((a, b) => (b.count > a.count ? b : a)).emotion
    : '—';
  const avgConfidence = checkins.length
    ? checkins.reduce((sum, c) => sum + c.confidence, 0) / checkins.length
    : 0;

  return (
    <Layout>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-bold text-teal-900">📈 History &amp; Trends</h1>
        <select
          value={rangeIdx}
          onChange={(e) => setRangeIdx(Number(e.target.value))}
          className="rounded-lg border border-teal-200 px-3 py-1.5 text-sm text-teal-900 outline-none focus:border-teal-500"
        >
          {RANGE_OPTIONS.map((o, i) => (
            <option key={o.label} value={i}>{o.label}</option>
          ))}
        </select>
      </div>

      {loading ? (
        <p className="mt-8 text-teal-900/50">Loading...</p>
      ) : checkins.length === 0 ? (
        <div className="mt-6 rounded-2xl border border-teal-100 bg-white p-8 text-center text-teal-900/60">
          No check-ins yet in this range. Head to Daily Mood Check-in to record one.
        </div>
      ) : (
        <>
          {/* Pattern section */}
          <section className="mt-6">
            <h2 className="text-lg font-semibold text-teal-900">Long-term mood patterns</h2>
            <p className="mt-1 text-sm text-teal-900/60">
              Computed from your actual check-in history — not a single reading, not a form. This
              is still a heuristic; treat it as a prompt to reflect, not a diagnosis.
            </p>

            {pattern?.insufficient_data ? (
              <div className="mt-4 rounded-2xl border border-teal-100 bg-teal-50/60 p-5 text-sm text-teal-900/80">
                You&rsquo;ve done {pattern.n_checkins} check-in(s) so far — need at least{' '}
                {pattern.min_required} before a pattern means anything. Keep checking in.
              </div>
            ) : pattern ? (
              <>
                <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <StatCard label="Check-ins analyzed" value={pattern.n_checkins} />
                  <StatCard label="Low-mood (Sad) %" value={`${(pattern.low_mood_pct * 100).toFixed(0)}%`} />
                  <StatCard label="Anxious (Fear) %" value={`${(pattern.anxious_pct * 100).toFixed(0)}%`} />
                  <StatCard label="Positive (Happy) %" value={`${(pattern.positive_pct * 100).toFixed(0)}%`} />
                </div>
                <p className="mt-3 text-sm font-medium text-teal-900">Trend: {pattern.trend}</p>

                {pattern.flags?.length > 0 ? (
                  <div className="mt-3 space-y-2">
                    {pattern.flags.map((f) => (
                      <div key={f.label} className="rounded-xl border border-amber-200 bg-amber-50 p-4">
                        <p className="font-semibold text-amber-900">⚠️ {f.label}</p>
                        <p className="mt-1 text-sm text-amber-900/80">{f.detail}</p>
                      </div>
                    ))}
                    <p className="text-xs text-teal-900/50">
                      If any of this reflects how you&rsquo;ve actually been feeling, consider talking to
                      a mental health professional or someone you trust — see the Resources page.
                    </p>
                  </div>
                ) : (
                  <div className="mt-3 rounded-xl bg-green-50 p-4 text-sm text-green-800">
                    ✅ No sustained concerning pattern detected in this window.
                  </div>
                )}
              </>
            ) : null}
          </section>

          {/* Summary stats */}
          <section className="mt-8">
            <h2 className="text-lg font-semibold text-teal-900">Check-in history</h2>
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <StatCard label="Check-ins in range" value={checkins.length} />
              <StatCard label="Current streak" value={`${streak} day${streak !== 1 ? 's' : ''}`} />
              <StatCard label="Most common emotion" value={mostCommon} />
              <StatCard label="Avg. confidence" value={`${(avgConfidence * 100).toFixed(0)}%`} />
            </div>
          </section>

          {/* Line chart */}
          <section className="mt-6 rounded-2xl border border-teal-100 bg-white p-5 shadow-sm">
            <h3 className="mb-3 text-sm font-semibold text-teal-900">Emotion probabilities over time</h3>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={lineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#EFF7F5" />
                <XAxis dataKey="time" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Legend />
                {emotionCols.map((e) => (
                  <Line key={e} type="monotone" dataKey={e} stroke={EMOTION_COLORS[e]} dot={{ r: 2 }} strokeWidth={2} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </section>

          {/* Bar chart */}
          <section className="mt-6 rounded-2xl border border-teal-100 bg-white p-5 shadow-sm">
            <h3 className="mb-3 text-sm font-semibold text-teal-900">Emotion distribution</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#EFF7F5" />
                <XAxis dataKey="emotion" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {distribution.map((d) => (
                    <Cell key={d.emotion} fill={EMOTION_COLORS[d.emotion]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </section>

          {/* Raw table */}
          <details className="mt-6 rounded-2xl border border-teal-100 bg-white p-5 shadow-sm">
            <summary className="cursor-pointer text-sm font-semibold text-teal-900">
              All check-ins in range (raw data)
            </summary>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-teal-100 text-teal-900/50">
                    <th className="py-2 pr-4">Date</th>
                    <th className="py-2 pr-4">Emotion</th>
                    <th className="py-2 pr-4">Confidence</th>
                    <th className="py-2 pr-4">Sentence</th>
                  </tr>
                </thead>
                <tbody>
                  {[...checkins].reverse().map((c) => (
                    <tr key={c.id} className="border-b border-teal-50">
                      <td className="py-2 pr-4 text-teal-900/70">
                        {new Date(c.created_at).toLocaleString()}
                      </td>
                      <td className="py-2 pr-4 font-medium" style={{ color: EMOTION_COLORS[c.predicted_emotion] }}>
                        {c.predicted_emotion}
                      </td>
                      <td className="py-2 pr-4 text-teal-900/70">{(c.confidence * 100).toFixed(0)}%</td>
                      <td className="py-2 pr-4 text-teal-900/50">{c.prompt_sentence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
    </Layout>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="rounded-xl border border-teal-100 bg-white p-4 text-center shadow-sm">
      <p className="text-xs text-teal-900/50">{label}</p>
      <p className="mt-1 text-lg font-bold text-teal-900">{value}</p>
    </div>
  );
}

function computeStreak(checkins) {
  const dates = new Set(checkins.map((c) => new Date(c.date).toDateString()));
  let streak = 0;
  const cursor = new Date();
  while (dates.has(cursor.toDateString())) {
    streak += 1;
    cursor.setDate(cursor.getDate() - 1);
  }
  return streak;
}

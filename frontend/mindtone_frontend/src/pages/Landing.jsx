import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Landing() {
  const [tab, setTab] = useState('login');
  const navigate = useNavigate();
  const { login, signup } = useAuth();

  const [loginForm, setLoginForm] = useState({ username: '', password: '' });
  const [signupForm, setSignupForm] = useState({ username: '', email: '', password: '', confirm: '' });
  const [error, setError] = useState('');
  const [signupSuccess, setSignupSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  async function handleLogin(e) {
    e.preventDefault();
    setError('');
    setLoading(true);
    const { error } = await login(loginForm.username, loginForm.password);
    setLoading(false);
    if (error) setError(error);
    else navigate('/dashboard');
  }

  async function handleSignup(e) {
    e.preventDefault();
    setError('');
    if (signupForm.password !== signupForm.confirm) {
      setError("Passwords don't match.");
      return;
    }
    setLoading(true);
    const { error } = await signup(signupForm.username, signupForm.email, signupForm.password);
    setLoading(false);
    if (error) setError(error);
    else navigate('/dashboard');
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-teal-50 to-white">
      <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
        {/* Hero */}
        <div className="animate-fade-in-up relative overflow-hidden rounded-3xl bg-gradient-to-br from-teal-500 to-teal-400 px-6 py-14 text-center shadow-xl shadow-teal-500/20 sm:py-20">
          <div className="pointer-events-none absolute -right-16 -top-16 h-64 w-64 rounded-full bg-white/10" />
          <div className="pointer-events-none absolute -bottom-24 -left-10 h-72 w-72 rounded-full bg-white/10" />
          <div className="relative">
            <span className="mb-4 inline-flex h-16 w-16 items-center justify-center rounded-full bg-white/20 text-3xl backdrop-blur-sm animate-pulse-ring">
              🎙️
            </span>
            <h1 className="text-4xl font-bold text-white sm:text-5xl">MindTone</h1>
            <p className="mx-auto mt-3 max-w-xl text-lg text-white/90">
              A voice check-in that tracks how you&rsquo;re really doing, over weeks &mdash; not just today.
            </p>
          </div>
        </div>

        {/* Feature cards */}
        <div className="mt-8 grid gap-4 sm:grid-cols-3">
          {[
            { icon: '🗣️', title: 'Speak', desc: 'Read a short sentence out loud — takes under a minute.' },
            { icon: '📊', title: 'Track', desc: 'Every check-in is logged. Nothing gets overwritten.' },
            { icon: '📈', title: 'See patterns', desc: 'Real long-term patterns from your own data, over weeks.' },
          ].map((f, i) => (
            <div
              key={f.title}
              className="animate-fade-in-up rounded-2xl border border-teal-100 bg-white p-5 text-center shadow-sm"
              style={{ animationDelay: `${i * 100}ms` }}
            >
              <div className="text-3xl">{f.icon}</div>
              <h3 className="mt-2 font-semibold text-teal-900">{f.title}</h3>
              <p className="mt-1 text-sm text-teal-900/60">{f.desc}</p>
            </div>
          ))}
        </div>

        {/* Auth card */}
        <div className="animate-fade-in-up mx-auto mt-10 max-w-md rounded-2xl border border-teal-100 bg-white p-6 shadow-lg shadow-teal-900/5">
          <div className="mb-5 flex rounded-full bg-teal-50 p-1">
            {['login', 'signup'].map((t) => (
              <button
                key={t}
                onClick={() => { setTab(t); setError(''); }}
                className={`flex-1 rounded-full py-2 text-sm font-medium capitalize transition-colors ${
                  tab === t ? 'bg-white text-teal-900 shadow-sm' : 'text-teal-900/50'
                }`}
              >
                {t === 'login' ? 'Log in' : 'Sign up'}
              </button>
            ))}
          </div>

          {error && (
            <div className="mb-4 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>
          )}
          {signupSuccess && (
            <div className="mb-4 rounded-lg bg-green-50 px-3 py-2 text-sm text-green-700">
              Account created — you can log in now.
            </div>
          )}

          {tab === 'login' ? (
            <form onSubmit={handleLogin} className="space-y-3">
              <input
                type="text"
                placeholder="Username"
                required
                value={loginForm.username}
                onChange={(e) => setLoginForm({ ...loginForm, username: e.target.value })}
                className="w-full rounded-lg border border-teal-200 px-3 py-2 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20"
              />
              <input
                type="password"
                placeholder="Password"
                required
                value={loginForm.password}
                onChange={(e) => setLoginForm({ ...loginForm, password: e.target.value })}
                className="w-full rounded-lg border border-teal-200 px-3 py-2 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20"
              />
              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-lg bg-teal-500 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-teal-600 disabled:opacity-60"
              >
                {loading ? 'Logging in...' : 'Log in'}
              </button>
            </form>
          ) : (
            <form onSubmit={handleSignup} className="space-y-3">
              <input
                type="text"
                placeholder="Choose a username"
                required
                value={signupForm.username}
                onChange={(e) => setSignupForm({ ...signupForm, username: e.target.value })}
                className="w-full rounded-lg border border-teal-200 px-3 py-2 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20"
              />
              <input
                type="email"
                placeholder="Email"
                required
                value={signupForm.email}
                onChange={(e) => setSignupForm({ ...signupForm, email: e.target.value })}
                className="w-full rounded-lg border border-teal-200 px-3 py-2 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20"
              />
              <input
                type="password"
                placeholder="Password (8+ chars, letter + number)"
                required
                value={signupForm.password}
                onChange={(e) => setSignupForm({ ...signupForm, password: e.target.value })}
                className="w-full rounded-lg border border-teal-200 px-3 py-2 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20"
              />
              <input
                type="password"
                placeholder="Confirm password"
                required
                value={signupForm.confirm}
                onChange={(e) => setSignupForm({ ...signupForm, confirm: e.target.value })}
                className="w-full rounded-lg border border-teal-200 px-3 py-2 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20"
              />
              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-lg bg-teal-500 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-teal-600 disabled:opacity-60"
              >
                {loading ? 'Creating account...' : 'Create account'}
              </button>
            </form>
          )}
        </div>

        <p className="mx-auto mt-6 max-w-md text-center text-xs text-teal-900/50">
          Mood is tracked from your speech over time — patterns are derived from your own
          accumulated data, not a single reading or a form. This isn&rsquo;t a diagnosis.
        </p>
      </div>
    </div>
  );
}

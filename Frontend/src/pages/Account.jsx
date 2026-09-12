import { useEffect, useState } from 'react';
import Layout from '../components/Layout';
import { api } from '../api/client';

export default function Account() {
  const [profile, setProfile] = useState(null);
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [message, setMessage] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    api.getMe().then(setProfile).catch(() => {});
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    setMessage(null);
    if (newPassword !== confirmPassword) {
      setMessage({ type: 'error', text: "New passwords don't match." });
      return;
    }
    setSubmitting(true);
    const { error } = await api.changePassword(currentPassword, newPassword);
    setSubmitting(false);
    if (error) {
      setMessage({ type: 'error', text: error });
    } else {
      setMessage({ type: 'success', text: 'Password updated.' });
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    }
  }

  return (
    <Layout>
      <h1 className="text-2xl font-bold text-teal-900">⚙️ Account</h1>

      {profile && (
        <div className="mt-4 rounded-2xl border border-teal-100 bg-white p-6 shadow-sm">
          <p className="text-sm text-teal-900/80"><strong>Username:</strong> {profile.username}</p>
          <p className="mt-1 text-sm text-teal-900/80"><strong>Email:</strong> {profile.email}</p>
          <p className="mt-1 text-sm text-teal-900/80">
            <strong>Member since:</strong>{' '}
            {new Date(profile.created_at).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })}
          </p>
        </div>
      )}

      <div className="mt-4 rounded-2xl border border-teal-100 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-teal-900">Change password</h2>

        {message && (
          <div className={`mt-3 rounded-lg px-3 py-2 text-sm ${
            message.type === 'error' ? 'bg-red-50 text-red-700' : 'bg-green-50 text-green-700'
          }`}>
            {message.text}
          </div>
        )}

        <form onSubmit={handleSubmit} className="mt-4 space-y-3">
          <input
            type="password"
            placeholder="Current password"
            required
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            className="w-full rounded-lg border border-teal-200 px-3 py-2 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20"
          />
          <input
            type="password"
            placeholder="New password"
            required
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            className="w-full rounded-lg border border-teal-200 px-3 py-2 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20"
          />
          <input
            type="password"
            placeholder="Confirm new password"
            required
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            className="w-full rounded-lg border border-teal-200 px-3 py-2 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20"
          />
          <button
            type="submit"
            disabled={submitting}
            className="rounded-lg bg-teal-500 px-5 py-2.5 text-sm font-semibold text-white hover:bg-teal-600 disabled:opacity-60"
          >
            {submitting ? 'Updating...' : 'Update password'}
          </button>
        </form>
      </div>
    </Layout>
  );
}

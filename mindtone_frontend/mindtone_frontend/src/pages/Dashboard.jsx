import { Link } from 'react-router-dom';
import Layout from '../components/Layout';
import { useAuth } from '../context/AuthContext';

export default function Dashboard() {
  const { username } = useAuth();

  const cards = [
    { to: '/checkin', icon: '🎙️', title: 'Daily Mood', desc: 'Read a sentence, track your detected emotion.', cta: 'Check in' },
    { to: '/history', icon: '📈', title: 'History & Trends', desc: 'See your long-term patterns.', cta: 'View History' },
    { to: '/resources', icon: '🤝', title: 'Resources', desc: 'Support resources, always available.', cta: 'View Resources' },
  ];

  return (
    <Layout>
      <h1 className="animate-fade-in-up text-2xl font-bold text-teal-900 sm:text-3xl">
        Welcome back, {username} 👋
      </h1>

      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        {cards.map((c, i) => (
          <Link
            to={c.to}
            key={c.to}
            className="animate-fade-in-up group rounded-2xl border border-teal-100 bg-white p-6 shadow-sm transition-all hover:-translate-y-1 hover:shadow-lg hover:shadow-teal-500/10"
            style={{ animationDelay: `${i * 80}ms` }}
          >
            <div className="text-3xl">{c.icon}</div>
            <h3 className="mt-3 font-semibold text-teal-900">{c.title}</h3>
            <p className="mt-1 text-sm text-teal-900/60">{c.desc}</p>
            <span className="mt-3 inline-block text-sm font-medium text-teal-500 group-hover:text-teal-600">
              {c.cta} →
            </span>
          </Link>
        ))}
      </div>

      <div className="animate-fade-in-up mt-6 rounded-2xl border border-teal-100 bg-teal-50/60 p-5 text-sm text-teal-900/80">
        <strong>Note:</strong> long-term mood patterns here are derived heuristically from your
        own speech data, not a clinical diagnosis. If you&rsquo;re struggling, please talk to a
        mental health professional or a trusted person.
      </div>
    </Layout>
  );
}

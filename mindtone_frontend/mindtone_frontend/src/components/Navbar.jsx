import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const navItems = [
  { to: '/dashboard', label: 'Home' },
  { to: '/checkin', label: 'Daily Mood' },
  { to: '/history', label: 'History & Trends' },
  { to: '/resources', label: 'Resources' },
  { to: '/account', label: 'Account' },
];

export default function Navbar() {
  const { logout, username } = useAuth();
  const navigate = useNavigate();

  return (
    <nav className="sticky top-0 z-20 border-b border-teal-100 bg-white/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <div className="flex items-center gap-2">
          <span className="text-xl">🎙️</span>
          <span className="font-semibold text-teal-900">MindTone</span>
        </div>

        <div className="hidden gap-1 md:flex">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-teal-500 text-white'
                    : 'text-teal-900/70 hover:bg-teal-50 hover:text-teal-900'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <span className="hidden text-sm text-teal-900/60 sm:inline">Hi, {username}</span>
          <button
            onClick={() => {
              logout();
              navigate('/');
            }}
            className="rounded-full border border-teal-200 px-3 py-1.5 text-sm font-medium text-teal-700 transition-colors hover:bg-teal-50"
          >
            Log out
          </button>
        </div>
      </div>

      {/* Mobile nav */}
      <div className="flex gap-1 overflow-x-auto border-t border-teal-50 px-4 py-2 md:hidden">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `whitespace-nowrap rounded-full px-3 py-1 text-xs font-medium ${
                isActive ? 'bg-teal-500 text-white' : 'bg-teal-50 text-teal-900/70'
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}

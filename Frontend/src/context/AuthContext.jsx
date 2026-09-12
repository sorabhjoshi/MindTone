import { createContext, useContext, useState, useCallback } from 'react';
import { api } from '../api/client';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem('mindtone_token'));
  const [username, setUsername] = useState(() => localStorage.getItem('mindtone_username'));

  const login = useCallback(async (usernameInput, password) => {
    const { data, error } = await api.login(usernameInput, password);
    if (error) return { error };
    localStorage.setItem('mindtone_token', data.access_token);
    localStorage.setItem('mindtone_username', data.username);
    setToken(data.access_token);
    setUsername(data.username);
    return { error: null };
  }, []);

  const signup = useCallback(async (usernameInput, email, password) => {
    const { data, error } = await api.signup(usernameInput, email, password);
    if (error) return { error };
    localStorage.setItem('mindtone_token', data.access_token);
    localStorage.setItem('mindtone_username', data.username);
    setToken(data.access_token);
    setUsername(data.username);
    return { error: null };
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('mindtone_token');
    localStorage.removeItem('mindtone_username');
    setToken(null);
    setUsername(null);
  }, []);

  return (
    <AuthContext.Provider value={{ token, username, isAuthenticated: !!token, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}

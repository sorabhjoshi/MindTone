import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const client = axios.create({ baseURL: API_BASE_URL });

client.interceptors.request.use((config) => {
  const token = localStorage.getItem('mindtone_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Centralize the "expired/invalid token -> log out" behavior in one
// place, rather than every page having to check for a 401 itself.
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('mindtone_token');
      localStorage.removeItem('mindtone_username');
      if (!window.location.pathname.startsWith('/login')) {
        window.location.href = '/';
      }
    }
    return Promise.reject(error);
  }
);

function extractErrorMessage(error) {
  const detail = error.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg;
  return 'Something went wrong. Please try again.';
}

export const api = {
  async signup(username, email, password) {
    try {
      const { data } = await client.post('/api/auth/signup', { username, email, password });
      return { data, error: null };
    } catch (e) {
      return { data: null, error: extractErrorMessage(e) };
    }
  },

  async login(username, password) {
    try {
      const { data } = await client.post('/api/auth/login', { username, password });
      return { data, error: null };
    } catch (e) {
      return { data: null, error: extractErrorMessage(e) };
    }
  },

  async getMe() {
    const { data } = await client.get('/api/account/me');
    return data;
  },

  async changePassword(currentPassword, newPassword) {
    try {
      await client.post('/api/account/change-password', {
        current_password: currentPassword,
        new_password: newPassword,
      });
      return { error: null };
    } catch (e) {
      return { error: extractErrorMessage(e) };
    }
  },

  async getPrompt(exclude) {
    const { data } = await client.get('/api/checkin/prompt', { params: { exclude } });
    return data.sentence;
  },

  async submitCheckin(promptSentence, audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'checkin.webm');
    try {
      const { data } = await client.post('/api/checkin', formData, {
        params: { prompt_sentence: promptSentence },
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return { data, error: null };
    } catch (e) {
      return { data: null, error: extractErrorMessage(e) };
    }
  },

  async getTodayCheckins() {
    const { data } = await client.get('/api/checkin/today');
    return data;
  },

  async getHistory(days) {
    const { data } = await client.get('/api/checkin/history', { params: { days } });
    return data;
  },

  async getPatterns(days) {
    const { data } = await client.get('/api/patterns', { params: { days } });
    return data;
  },
};

export default client;

const API_BASE = '/api';

async function request(endpoint, options = {}) {
  const token = localStorage.getItem('pensieve_token');
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const response = await fetch(`${API_BASE}${endpoint}`, {
    cache: 'no-store',
    ...options,
    headers,
  });
  if (response.status === 401) {
    localStorage.removeItem('pensieve_token');
    localStorage.removeItem('pensieve_user');
    window.location.href = '/login';
    throw new Error('Session expired');
  }
  if (response.status === 204) return null;
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || JSON.stringify(error));
  }
  return response.json();
}

// Auth
export const authAPI = {
  register: (data) => request('/auth/register', { method: 'POST', body: JSON.stringify(data) }),
  login: (data) => request('/auth/login', { method: 'POST', body: JSON.stringify(data) }),
  getMe: () => request('/auth/me'),
};

// Entries
export const entriesAPI = {
  list: (params = '') => request(`/entries${params ? '?' + params : ''}`),
  get: (id) => request(`/entries/${id}`),
  create: (data) => request('/entries', { method: 'POST', body: JSON.stringify(data) }),
  update: (id, data) => request(`/entries/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
  delete: (id) => request(`/entries/${id}`, { method: 'DELETE' }),
  autosave: (data) => request('/entries/autosave', { method: 'POST', body: JSON.stringify(data) }),
};

// Analysis
export const analysisAPI = {
  analyze: (entryId) => request(`/analyze/${entryId}`, { method: 'POST' }),
  patterns: () => request('/patterns'),
};

// Reflections
export const reflectionsAPI = {
  suggest: () => request('/reflections/suggest', { method: 'POST' }),
  list: () => request('/reflections'),
  get: (id) => request(`/reflections/${id}`),
};

// Concepts
export const conceptsAPI = {
  list: (params = '') => request(`/concepts${params ? '?' + params : ''}`),
  get: (id) => request(`/concepts/${id}`),
};

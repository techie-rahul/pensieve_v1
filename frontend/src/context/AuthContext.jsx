import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { authAPI } from '../api/client';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('pensieve_token'));
  const [isLoading, setIsLoading] = useState(true);

  const isAuthenticated = !!token && !!user;

  useEffect(() => {
    if (token) {
      authAPI.getMe()
        .then(userData => {
          setUser(userData);
          setIsLoading(false);
        })
        .catch(() => {
          localStorage.removeItem('pensieve_token');
          setToken(null);
          setUser(null);
          setIsLoading(false);
        });
    } else {
      setIsLoading(false);
    }
  }, [token]);

  const login = useCallback(async (email, password) => {
    const data = await authAPI.login({ email, password });
    localStorage.setItem('pensieve_token', data.access_token);
    setToken(data.access_token);
    const userData = await authAPI.getMe();
    setUser(userData);
    return userData;
  }, []);

  const register = useCallback(async (email, password, name) => {
    await authAPI.register({ email, password, name });
    return login(email, password);
  }, [login]);

  const logout = useCallback(() => {
    localStorage.removeItem('pensieve_token');
    setToken(null);
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, token, isAuthenticated, isLoading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}

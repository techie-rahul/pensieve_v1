import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(email, password);
      navigate('/');
    } catch (err) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-ivory flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="font-serif text-3xl font-medium text-charcoal mb-2">Pensieve</h1>
          <p className="text-muted text-sm">Your journal stays private to your account.</p>
        </div>
        <form onSubmit={handleSubmit} className="bg-surface border border-border rounded-xl p-8 shadow-sm">
          <h2 className="font-serif text-xl text-charcoal mb-6">Welcome back</h2>
          {error && <div className="text-red-600 text-sm bg-red-50 border border-red-200 rounded-lg px-4 py-3 mb-4">{error}</div>}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-charcoal mb-1.5">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full px-4 py-3 bg-ivory border border-border rounded-lg text-charcoal placeholder:text-muted/50 focus:border-forest focus:ring-1 focus:ring-forest/20 transition-colors"
                placeholder="you@example.com"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-charcoal mb-1.5">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full px-4 py-3 bg-ivory border border-border rounded-lg text-charcoal placeholder:text-muted/50 focus:border-forest focus:ring-1 focus:ring-forest/20 transition-colors"
                placeholder="Enter your password"
              />
            </div>
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full mt-6 py-3 bg-forest text-white rounded-lg hover:bg-forest-dark transition-colors font-medium disabled:opacity-50"
          >
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
          <p className="text-center text-sm text-muted mt-4">
            Don't have an account?{' '}
            <Link to="/register" className="text-forest hover:underline">Create one</Link>
          </p>
        </form>
      </div>
    </div>
  );
}

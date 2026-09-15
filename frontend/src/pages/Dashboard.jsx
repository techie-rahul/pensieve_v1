import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { entriesAPI, analysisAPI, reflectionsAPI } from '../api/client';
import { PenLine, ArrowRight, BookOpen } from 'lucide-react';
import EmptyState from '../components/common/EmptyState';

function getGreeting() {
  const h = new Date().getHours();
  if (h < 12) return 'Good morning';
  if (h < 18) return 'Good afternoon';
  return 'Good evening';
}

function formatDate(dateStr) {
  return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

export default function Dashboard() {
  const { user } = useAuth();
  const [entries, setEntries] = useState([]);
  const [patterns, setPatterns] = useState(null);
  const [reflections, setReflections] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([
      entriesAPI.list('limit=5&is_draft=false'),
      analysisAPI.patterns(),
      reflectionsAPI.list(),
    ]).then(([entriesRes, patternsRes, reflectionsRes]) => {
      if (entriesRes.status === 'fulfilled') setEntries(entriesRes.value);
      if (patternsRes.status === 'fulfilled') setPatterns(patternsRes.value);
      if (reflectionsRes.status === 'fulfilled') setReflections(reflectionsRes.value);
      setLoading(false);
    });
  }, []);

  const getPatternObservations = () => {
    if (!patterns || (patterns.status !== 'success' && patterns.status !== 'ready')) return [];
    const obs = [];
    const trends = [patterns.emotion_trends, patterns.theme_trends, patterns.linguistic_trends];
    for (const t of trends) {
      if (t && typeof t === 'object' && Array.isArray(t.observations)) {
        obs.push(...t.observations);
      }
    }
    return obs.slice(0, 4);
  };

  if (loading) {
    return <div className="text-muted py-12 text-center font-serif">Loading your journal...</div>;
  }

  const patternObs = getPatternObservations();
  const latestReflection = reflections.length > 0 ? reflections[0] : null;

  return (
    <div className="space-y-10">
      {/* Greeting */}
      <div>
        <h1 className="font-serif text-2xl md:text-3xl text-charcoal mb-1">
          {getGreeting()}, {user?.name?.split(' ')[0] || 'there'}
        </h1>
        <p className="text-muted text-sm">Take a moment to reflect on your day.</p>
        <Link
          to="/journal"
          className="inline-flex items-center gap-2 mt-4 px-6 py-3 bg-forest text-white rounded-lg hover:bg-forest-dark transition-colors text-sm font-medium"
        >
          <PenLine className="w-4 h-4" />
          Continue writing
        </Link>
      </div>

      {/* Recent Entries */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-serif text-lg text-charcoal">Recent entries</h2>
          <Link to="/history" className="text-sm text-forest hover:underline flex items-center gap-1">
            View all <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
        {entries.length === 0 ? (
          <EmptyState
            icon={BookOpen}
            title="No entries yet"
            description="Start writing to see your journal entries here."
          />
        ) : (
          <div className="space-y-1">
            {entries.map((entry) => (
              <Link
                key={entry.id}
                to={`/journal?id=${entry.id}`}
                className="block py-3 px-4 -mx-4 rounded-lg hover:bg-surface transition-colors group"
              >
                <div className="flex items-baseline justify-between mb-1">
                  <h3 className="text-sm font-medium text-charcoal group-hover:text-forest transition-colors truncate">
                    {entry.title || 'Untitled'}
                  </h3>
                  <span className="text-xs text-muted ml-4 flex-shrink-0">{formatDate(entry.created_at)}</span>
                </div>
                <p className="text-xs text-muted line-clamp-1">
                  {entry.content?.substring(0, 120) || ''}
                </p>
              </Link>
            ))}
          </div>
        )}
      </section>

      {/* Patterns */}
      <section>
        <h2 className="font-serif text-lg text-charcoal mb-4">Your patterns</h2>
        {patternObs.length > 0 ? (
          <div className="space-y-3">
            {patternObs.map((obs, i) => (
              <div key={i} className="flex gap-3 items-start">
                <div className="w-1.5 h-1.5 rounded-full bg-sage mt-2 flex-shrink-0" />
                <p className="text-sm text-muted">{typeof obs === 'string' ? obs : obs.description || obs.observation || JSON.stringify(obs)}</p>
              </div>
            ))}
            <Link to="/patterns" className="text-sm text-forest hover:underline inline-flex items-center gap-1 mt-2">
              See full patterns <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
        ) : (
          <p className="text-sm text-muted bg-surface border border-border rounded-lg px-4 py-3">
            Write a few more entries to uncover patterns. Longitudinal analysis requires at least 3 entries over 7 or more days.
          </p>
        )}
      </section>

      {/* Recent Reflection */}
      {latestReflection && (
        <section>
          <h2 className="font-serif text-lg text-charcoal mb-4">Recent reflection</h2>
          <div className="bg-surface border border-border rounded-lg px-5 py-4">
            <p className="text-sm text-charcoal leading-relaxed">
              {latestReflection.reflection_text?.substring(0, 200)}
              {latestReflection.reflection_text?.length > 200 ? '...' : ''}
            </p>
            <Link to="/reflections" className="text-sm text-forest hover:underline mt-3 inline-block">
              View all reflections
            </Link>
          </div>
        </section>
      )}
    </div>
  );
}

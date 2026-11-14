import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { analysisAPI } from '../api/client';
import { TrendingUp, PenLine, RotateCcw, AlertCircle } from 'lucide-react';

export default function Patterns() {
  const [patterns, setPatterns] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchPatterns = useCallback(() => {
    setLoading(true);
    setError('');
    analysisAPI
      .patterns()
      .then((data) => {
        setPatterns(data);
      })
      .catch((err) => {
        const message =
          err.message && err.message !== 'Request failed'
            ? err.message
            : 'Unable to synthesize longitudinal patterns across your entries. The pattern synthesis service may be unavailable.';
        setError(message);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchPatterns();
  }, [fetchPatterns]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[360px]">
        <div className="text-muted/80 font-serif text-lg tracking-wide animate-pulse">
          Gathering your patterns...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <div className="mb-6">
          <h1 className="font-serif text-2xl text-charcoal mb-1">Patterns</h1>
          <p className="text-sm text-muted">
            Longitudinal observations and trends across your writing history.
          </p>
        </div>

        <div className="bg-surface border border-border rounded-xl p-8 text-center max-w-2xl mx-auto my-6 shadow-2xs">
          <div className="w-12 h-12 rounded-full bg-red-50 border border-red-200 flex items-center justify-center mx-auto mb-4 text-red-700">
            <AlertCircle className="w-6 h-6" strokeWidth={1.5} />
          </div>

          <h2 className="font-serif text-xl text-charcoal mb-2">Unable to load reflection patterns</h2>
          <p className="text-sm text-muted max-w-md mx-auto mb-6 leading-relaxed">
            {error}
          </p>

          <div className="flex items-center justify-center gap-3">
            <button
              onClick={fetchPatterns}
              className="inline-flex items-center gap-2 px-4 py-2 bg-forest text-white rounded-lg hover:bg-forest-dark transition-colors text-xs font-medium cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Try again</span>
            </button>
            <Link
              to="/journal"
              className="inline-flex items-center gap-2 px-4 py-2 bg-surface border border-border text-charcoal rounded-lg hover:bg-ivory transition-colors text-xs font-medium"
            >
              Return to Journal
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const isSuccess = patterns && (patterns.status === 'success' || patterns.status === 'ready');

  // Empty state for insufficient_data, sparse_history, or any non-success status
  if (!isSuccess) {
    const entryCount = patterns?.entry_count || 0;
    const timeSpan = patterns?.time_span_days;
    const displaySpan =
      typeof timeSpan === 'number' ? Math.round(timeSpan) : timeSpan;

    return (
      <div>
        <div className="mb-6">
          <h1 className="font-serif text-2xl text-charcoal mb-1">Patterns</h1>
          <p className="text-sm text-muted">
            Longitudinal observations and trends across your writing history.
          </p>
        </div>

        <div className="bg-surface border border-border rounded-xl p-8 text-center max-w-2xl mx-auto my-6 shadow-2xs">
          <div className="w-12 h-12 rounded-full bg-cream flex items-center justify-center mx-auto mb-4 text-forest">
            <TrendingUp className="w-6 h-6" strokeWidth={1.5} />
          </div>

          <h2 className="font-serif text-xl text-charcoal mb-2">Not enough entries yet</h2>
          <p className="text-sm text-muted max-w-md mx-auto mb-6 leading-relaxed">
            Write at least 3 published entries spanning 7+ days to uncover longitudinal patterns.
          </p>

          {/* Progress / Status metrics */}
          <div className="inline-flex flex-wrap items-center justify-center gap-3 bg-ivory border border-border rounded-lg px-4 py-2.5 mb-6 text-xs text-muted">
            <span>
              <strong className="text-charcoal font-medium">{entryCount}</strong> / 3 published entries
            </span>
            {displaySpan !== null && displaySpan !== undefined && (
              <>
                <span className="text-border">•</span>
                <span>
                  <strong className="text-charcoal font-medium">{displaySpan}</strong>{' '}
                  {displaySpan === 1 ? 'day' : 'days'} span
                </span>
              </>
            )}
          </div>

          {patterns?.message && (
            <p className="text-xs text-muted/80 max-w-md mx-auto mb-6 italic">
              {patterns.message}
            </p>
          )}

          <div>
            <Link
              to="/journal"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-forest text-white rounded-lg hover:bg-forest-dark transition-colors text-sm font-medium"
            >
              <PenLine className="w-4 h-4" />
              Write an entry
            </Link>
          </div>
        </div>
      </div>
    );
  }

  // Helper renderers for successful data
  const renderObservations = (observations) => {
    if (!Array.isArray(observations) || observations.length === 0) return null;
    return (
      <div className="space-y-3 mt-3">
        {observations.map((obs, i) => (
          <div key={i} className="flex gap-3 items-start">
            <div className="w-1.5 h-1.5 rounded-full bg-sage mt-2 flex-shrink-0" />
            <p className="text-sm text-muted leading-relaxed">
              {typeof obs === 'string'
                ? obs
                : obs.description || obs.observation || JSON.stringify(obs)}
            </p>
          </div>
        ))}
      </div>
    );
  };

  const renderTopItems = (items) => {
    if (!Array.isArray(items) || items.length === 0) return null;
    return (
      <div className="flex flex-wrap gap-2 mt-3">
        {items.slice(0, 6).map((item, i) => {
          const name =
            typeof item === 'string'
              ? item
              : item.emotion || item.theme || item.name || item.label || JSON.stringify(item);
          const score =
            typeof item === 'object'
              ? item.score || item.count || item.frequency
              : null;
          return (
            <span
              key={i}
              className="inline-flex items-center gap-1 px-3 py-1.5 bg-cream text-charcoal rounded-lg text-xs"
            >
              <span className="capitalize">{name}</span>
              {score != null && (
                <span className="text-muted">
                  (
                  {typeof score === 'number' && score < 1
                    ? `${(score * 100).toFixed(0)}%`
                    : score}
                  )
                </span>
              )}
            </span>
          );
        })}
      </div>
    );
  };

  const hasEmotionData =
    patterns.emotion_trends &&
    ((Array.isArray(patterns.emotion_trends.top_emotions) &&
      patterns.emotion_trends.top_emotions.length > 0) ||
      (Array.isArray(patterns.emotion_trends.observations) &&
        patterns.emotion_trends.observations.length > 0));

  const hasThemeData =
    patterns.theme_trends &&
    ((Array.isArray(patterns.theme_trends.top_themes) &&
      patterns.theme_trends.top_themes.length > 0) ||
      (Array.isArray(patterns.theme_trends.observations) &&
        patterns.theme_trends.observations.length > 0));

  const hasLinguisticData =
    patterns.linguistic_trends &&
    Array.isArray(patterns.linguistic_trends.observations) &&
    patterns.linguistic_trends.observations.length > 0;

  const hasLexicalData =
    patterns.recurring_lexical_patterns &&
    Array.isArray(patterns.recurring_lexical_patterns.observations) &&
    patterns.recurring_lexical_patterns.observations.length > 0;

  const displayTimeSpan =
    typeof patterns.time_span_days === 'number'
      ? Math.round(patterns.time_span_days)
      : patterns.time_span_days;

  return (
    <div>
      <div className="mb-8">
        <h1 className="font-serif text-2xl text-charcoal mb-1">Patterns</h1>
        <p className="text-sm text-muted">
          Observations from {patterns.entry_count || 0} journal entries
          {displayTimeSpan != null ? ` spanning ${displayTimeSpan} days` : ''}.
        </p>
      </div>

      <div className="space-y-8">
        {/* Emotional Patterns */}
        {hasEmotionData && (
          <section className="bg-surface border border-border rounded-xl px-6 py-6 shadow-2xs">
            <h2 className="font-serif text-lg text-charcoal mb-1">Emotional patterns</h2>
            <p className="text-xs text-muted">Based on GoEmotions-based emotion analysis</p>
            {renderTopItems(patterns.emotion_trends.top_emotions)}
            {renderObservations(patterns.emotion_trends.observations)}
          </section>
        )}

        {/* Theme Patterns */}
        {hasThemeData && (
          <section className="bg-surface border border-border rounded-xl px-6 py-6 shadow-2xs">
            <h2 className="font-serif text-lg text-charcoal mb-1">Theme patterns</h2>
            <p className="text-xs text-muted">Themes discovered through semantic clustering</p>
            {renderTopItems(patterns.theme_trends.top_themes)}
            {renderObservations(patterns.theme_trends.observations)}
          </section>
        )}

        {/* Linguistic Patterns */}
        {hasLinguisticData && (
          <section className="bg-surface border border-border rounded-xl px-6 py-6 shadow-2xs">
            <h2 className="font-serif text-lg text-charcoal mb-1">Linguistic patterns</h2>
            <p className="text-xs text-muted">Writing style observations</p>
            {renderObservations(patterns.linguistic_trends.observations)}
          </section>
        )}

        {/* Recurring Lexical Patterns */}
        {hasLexicalData && (
          <section className="bg-surface border border-border rounded-xl px-6 py-6 shadow-2xs">
            <h2 className="font-serif text-lg text-charcoal mb-1">Recurring patterns</h2>
            <p className="text-xs text-muted">Lexical patterns observed across your entries</p>
            {renderObservations(patterns.recurring_lexical_patterns.observations)}
          </section>
        )}
      </div>
    </div>
  );
}

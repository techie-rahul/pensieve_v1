import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { analysisAPI } from '../api/client';
import {
  TrendingUp,
  PenLine,
  RotateCcw,
  AlertCircle,
  Calendar,
  ChevronDown,
  ChevronUp,
  BookOpen,
  Heart,
  Sparkles,
  Clock,
  Compass,
  MessageSquareQuote,
  Layers,
  ArrowRight,
  ExternalLink,
} from 'lucide-react';

export default function Patterns() {
  const [patterns, setPatterns] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [expandedEvidence, setExpandedEvidence] = useState({});

  const toggleEvidence = (id) => {
    setExpandedEvidence((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

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
      <div className="flex flex-col items-center justify-center min-h-[400px]">
        <div className="w-10 h-10 rounded-full border-2 border-forest/20 border-t-forest animate-spin mb-4" />
        <div className="text-charcoal font-serif text-lg tracking-wide">
          Reading your journal reflections...
        </div>
        <p className="text-xs text-muted mt-2">
          Discovering recurring topics, emotional trajectories, and language patterns.
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <div className="mb-6">
          <h1 className="font-serif text-3xl text-charcoal mb-1">Your Patterns</h1>
          <p className="text-sm text-muted">
            Insights discovered across your journal entries.
          </p>
        </div>

        <div className="bg-surface border border-border rounded-xl p-8 text-center max-w-2xl mx-auto my-6 shadow-2xs">
          <div className="w-12 h-12 rounded-full bg-red-50 border border-red-200 flex items-center justify-center mx-auto mb-4 text-red-700">
            <AlertCircle className="w-6 h-6" strokeWidth={1.5} />
          </div>

          <h2 className="font-serif text-xl text-charcoal mb-2">Unable to load patterns</h2>
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

  const isSuccess = patterns && (patterns.status === 'success' || patterns.status === 'ready' || patterns.status === 'ok');

  // Empty state for insufficient_data or sparse_history
  if (!isSuccess) {
    const entryCount = patterns?.entry_count || 0;
    const timeSpan = patterns?.time_span_days;
    const displaySpan = typeof timeSpan === 'number' ? Math.round(timeSpan) : timeSpan;

    return (
      <div>
        <div className="mb-6">
          <h1 className="font-serif text-3xl text-charcoal mb-1">Your Patterns</h1>
          <p className="text-sm text-muted">
            Insights discovered across your journal entries.
          </p>
        </div>

        <div className="bg-surface border border-border rounded-2xl p-10 text-center max-w-2xl mx-auto my-8 shadow-2xs">
          <div className="w-14 h-14 rounded-full bg-cream flex items-center justify-center mx-auto mb-4 text-forest">
            <TrendingUp className="w-7 h-7" strokeWidth={1.5} />
          </div>

          <h2 className="font-serif text-2xl text-charcoal mb-2">Not enough entries yet</h2>
          <p className="text-sm text-muted max-w-md mx-auto mb-6 leading-relaxed">
            Pensieve requires at least 3 published entries spanning 7+ days to synthesize genuine longitudinal insights without jumping to conclusions.
          </p>

          <div className="inline-flex flex-wrap items-center justify-center gap-4 bg-ivory border border-border rounded-xl px-5 py-3 mb-6 text-xs text-muted">
            <span>
              <strong className="text-charcoal font-medium text-sm">{entryCount}</strong> / 3 published entries
            </span>
            {displaySpan !== null && displaySpan !== undefined && (
              <>
                <span className="text-border">•</span>
                <span>
                  <strong className="text-charcoal font-medium text-sm">{displaySpan}</strong>{' '}
                  {displaySpan === 1 ? 'day' : 'days'} span (7+ required)
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
              className="inline-flex items-center gap-2 px-6 py-2.5 bg-forest text-white rounded-lg hover:bg-forest-dark transition-colors text-sm font-medium shadow-2xs"
            >
              <PenLine className="w-4 h-4" />
              Write an entry
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const overview = patterns.overview || {};
  const recurringTopics = patterns.recurring_topics || [];
  const emotionalTrends = patterns.emotional_trends || [];
  const recurringThoughts = patterns.recurring_thoughts || [];
  const changesOverTime = patterns.changes_over_time || [];
  const personalObservations = patterns.personal_observations || [];

  const displayTimeSpan =
    typeof overview.time_span_days === 'number'
      ? Math.round(overview.time_span_days)
      : typeof patterns.time_span_days === 'number'
      ? Math.round(patterns.time_span_days)
      : patterns.time_span_days;

  const totalEntries = overview.entry_count || patterns.entry_count || 0;

  // Reusable component to render verifiable matching entries with navigation
  const renderEvidenceEntries = (matchingEntries, label = 'Supporting entries') => {
    if (!Array.isArray(matchingEntries) || matchingEntries.length === 0) return null;

    return (
      <div className="space-y-2 pt-1.5">
        <span className="font-medium text-charcoal block text-xs">{label}:</span>
        <div className="space-y-2">
          {matchingEntries.map((item, i) => (
            <div key={i} className="pl-3 border-l-2 border-sage/60 py-1 bg-surface/50 rounded-r-md pr-2">
              <div className="flex items-center justify-between gap-2 text-2xs mb-0.5">
                <span className="font-semibold text-charcoal truncate">
                  {item.title || 'Journal Entry'}
                </span>
                <span className="text-muted flex-shrink-0">{item.date_short || item.date}</span>
              </div>
              <p className="italic text-charcoal/80 text-xs leading-relaxed">
                &ldquo;{item.excerpt}&rdquo;
              </p>
              {item.id && (
                <Link
                  to={`/journal?id=${item.id}`}
                  className="inline-flex items-center gap-1 text-forest hover:text-forest-dark text-2xs font-medium mt-1 transition-colors"
                >
                  <span>Open entry</span>
                  <ExternalLink className="w-2.5 h-2.5" />
                </Link>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-5xl mx-auto space-y-10 pb-16">
      {/* Editorial Header */}
      <div>
        <h1 className="font-serif text-3xl md:text-4xl text-charcoal mb-2 tracking-tight">Your Patterns</h1>
        <p className="text-base text-muted font-normal">
          Insights discovered across your journal entries.
        </p>

        {/* Overview Banner */}
        <div className="mt-6 bg-surface border border-border rounded-xl p-5 shadow-2xs flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-cream text-forest flex items-center justify-center flex-shrink-0">
              <Compass className="w-5 h-5" strokeWidth={1.5} />
            </div>
            <div>
              <p className="text-xs uppercase tracking-wider text-muted font-medium">Longitudinal Synthesis</p>
              <p className="text-sm font-medium text-charcoal">
                Based on <span className="font-semibold text-forest">{totalEntries} published entries</span>
                {overview.date_range ? ` spanning ${overview.date_range}` : displayTimeSpan ? ` across ${displayTimeSpan} days` : ''}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs text-muted bg-ivory px-3.5 py-1.5 rounded-lg border border-border/80">
            <Calendar className="w-3.5 h-3.5 text-sage" />
            <span>{overview.num_windows || 1} analytical {(overview.num_windows || 1) === 1 ? 'window' : 'windows'} evaluated</span>
          </div>
        </div>
      </div>

      {/* 1. RECURRING TOPICS */}
      {recurringTopics.length > 0 && (
        <section className="space-y-4">
          <div className="border-b border-border pb-2 flex items-baseline justify-between">
            <div className="flex items-center gap-2.5">
              <Layers className="w-5 h-5 text-forest" strokeWidth={1.5} />
              <h2 className="font-serif text-xl md:text-2xl text-charcoal">Recurring Topics</h2>
            </div>
            <span className="text-xs text-muted">Phase 2 Semantic Clustering</span>
          </div>
          <p className="text-xs text-muted max-w-2xl">
            Semantic themes identified across your reflections, grounded directly in the text of your entries.
          </p>

          <div className="grid grid-cols-1 gap-4 mt-4">
            {recurringTopics.map((topic, idx) => {
              const evidenceId = `topic-${idx}`;
              const isExpanded = !!expandedEvidence[evidenceId];
              const evidence = topic.evidence || {};

              return (
                <div
                  key={idx}
                  className="bg-surface border border-border rounded-xl p-6 shadow-2xs hover:border-forest/30 transition-colors"
                >
                  <div className="flex flex-wrap items-start justify-between gap-3 mb-2">
                    <div>
                      <h3 className="font-serif text-lg text-charcoal font-medium">{topic.title}</h3>
                      <p className="text-xs text-sage font-medium mt-0.5">{topic.theme_name}</p>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-cream text-charcoal">
                        {topic.strength || `${topic.percentage}% of entries`}
                      </span>
                      <span className="inline-flex items-center text-xs text-muted bg-ivory px-2.5 py-1 rounded-md border border-border">
                        {topic.entry_count} {topic.entry_count === 1 ? 'entry' : 'entries'}
                      </span>
                    </div>
                  </div>

                  <p className="text-sm text-charcoal/90 leading-relaxed mb-4">
                    {topic.description}
                  </p>

                  {/* Salient Keywords */}
                  {Array.isArray(topic.salient_keywords) && topic.salient_keywords.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5 mb-4">
                      <span className="text-xs text-muted mr-1">Distinctive keywords:</span>
                      {topic.salient_keywords.map((kw, i) => (
                        <span key={i} className="text-xs px-2 py-0.5 bg-ivory text-charcoal/80 rounded border border-border/70">
                          {kw}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Collapsible Evidence Section */}
                  <div className="border-t border-border/60 pt-3">
                    <button
                      onClick={() => toggleEvidence(evidenceId)}
                      className="inline-flex items-center gap-1.5 text-xs text-forest hover:text-forest-dark font-medium cursor-pointer transition-colors"
                    >
                      <span>{isExpanded ? 'Hide supporting evidence' : 'View supporting evidence'}</span>
                      {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    </button>

                    {isExpanded && (
                      <div className="mt-3.5 p-4 bg-ivory/70 rounded-lg border border-border/80 space-y-3 text-xs">
                        <div className="flex flex-wrap gap-x-6 gap-y-1 text-muted">
                          <span>
                            <strong>Date range:</strong> {evidence.date_range || topic.date_range}
                          </span>
                          <span>
                            <strong>Entries evaluated:</strong> {evidence.entry_count} of {evidence.total_entries} ({topic.percentage}%)
                          </span>
                        </div>

                        {/* Clickable matching entries */}
                        {renderEvidenceEntries(evidence.matching_entries, 'Representative matching entries')}

                        <p className="text-2xs text-muted/80 italic pt-1">
                          Signal source: Measured via Sentence-BERT prototype similarity aligned with Phase 2 taxonomy.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* 2. EMOTIONAL TRENDS */}
      {emotionalTrends.length > 0 && (
        <section className="space-y-4">
          <div className="border-b border-border pb-2 flex items-baseline justify-between">
            <div className="flex items-center gap-2.5">
              <Heart className="w-5 h-5 text-forest" strokeWidth={1.5} />
              <h2 className="font-serif text-xl md:text-2xl text-charcoal">Emotional Trends</h2>
            </div>
            <span className="text-xs text-muted">Phase 1 GoEmotions Chronology</span>
          </div>
          <p className="text-xs text-muted max-w-2xl">
            Observable shifts in emotional tone across earlier versus recent periods. Probabilities reflect model signal intensity, not psychological certainty.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            {emotionalTrends.map((trend, idx) => {
              const evidenceId = `emotion-${idx}`;
              const isExpanded = !!expandedEvidence[evidenceId];
              const evidence = trend.evidence || {};
              const isDecreasing = trend.direction === 'decreased';
              const isIncreasing = trend.direction === 'increased';

              return (
                <div
                  key={idx}
                  className="bg-surface border border-border rounded-xl p-6 shadow-2xs hover:border-forest/30 transition-colors flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <h3 className="font-serif text-lg text-charcoal font-medium">{trend.title}</h3>
                      <span
                        className={`text-xs px-2.5 py-1 rounded-md font-medium capitalize flex-shrink-0 ${
                          isDecreasing
                            ? 'bg-amber-50 text-amber-800 border border-amber-200'
                            : isIncreasing
                            ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                            : 'bg-cream text-charcoal'
                        }`}
                      >
                        {trend.direction}
                      </span>
                    </div>

                    <p className="text-sm text-charcoal/90 leading-relaxed mb-4">
                      {trend.description}
                    </p>

                    {/* Timeline comparison pill */}
                    <div className="grid grid-cols-2 gap-2 bg-ivory p-3 rounded-lg border border-border/80 text-xs mb-4">
                      <div>
                        <span className="text-muted block text-2xs uppercase">Earlier Period</span>
                        <span className="font-medium text-charcoal">
                          Signal: {(trend.earlier_score * 100).toFixed(0)}%
                        </span>
                        <span className="text-muted/70 block text-2xs truncate">
                          {trend.earlier_period?.date_range || 'Early entries'}
                        </span>
                      </div>
                      <div className="border-l border-border pl-2.5">
                        <span className="text-muted block text-2xs uppercase">Recent Period</span>
                        <span className="font-medium text-charcoal">
                          Signal: {(trend.recent_score * 100).toFixed(0)}%
                        </span>
                        <span className="text-muted/70 block text-2xs truncate">
                          {trend.recent_period?.date_range || 'Recent entries'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Collapsible Evidence Section */}
                  <div className="border-t border-border/60 pt-3">
                    <button
                      onClick={() => toggleEvidence(evidenceId)}
                      className="inline-flex items-center gap-1.5 text-xs text-forest hover:text-forest-dark font-medium cursor-pointer transition-colors"
                    >
                      <span>{isExpanded ? 'Hide supporting evidence' : 'View supporting evidence'}</span>
                      {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    </button>

                    {isExpanded && (
                      <div className="mt-3.5 p-4 bg-ivory/70 rounded-lg border border-border/80 space-y-3 text-xs">
                        {evidence.measurement_note && (
                          <p className="text-muted leading-relaxed">{evidence.measurement_note}</p>
                        )}

                        {/* Clickable matching entries */}
                        {renderEvidenceEntries(evidence.matching_entries, 'Supporting entries for this emotion signal')}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* 3. RECURRING THOUGHTS AND LANGUAGE */}
      {recurringThoughts.length > 0 && (
        <section className="space-y-4">
          <div className="border-b border-border pb-2 flex items-baseline justify-between">
            <div className="flex items-center gap-2.5">
              <MessageSquareQuote className="w-5 h-5 text-forest" strokeWidth={1.5} />
              <h2 className="font-serif text-xl md:text-2xl text-charcoal">Recurring Thoughts & Language</h2>
            </div>
            <span className="text-xs text-muted">Lexical Synthesis</span>
          </div>
          <p className="text-xs text-muted max-w-2xl">
            Distinctive phrases and topics repeatedly explored across separate entries, filtered of conversational filler.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            {recurringThoughts.map((thought, idx) => {
              const evidenceId = `thought-${idx}`;
              const isExpanded = !!expandedEvidence[evidenceId];
              const evidence = thought.evidence || {};

              return (
                <div
                  key={idx}
                  className="bg-surface border border-border rounded-xl p-6 shadow-2xs hover:border-forest/30 transition-colors"
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <h3 className="font-serif text-lg text-charcoal font-medium">{thought.title}</h3>
                    <span className="text-xs bg-cream text-charcoal px-2.5 py-1 rounded-md font-medium flex-shrink-0">
                      {thought.document_frequency} entries
                    </span>
                  </div>

                  <p className="text-sm text-charcoal/90 leading-relaxed mb-4">
                    {thought.description}
                  </p>

                  {/* Related Expressions (Semantic grouping) */}
                  {Array.isArray(thought.related_expressions) && thought.related_expressions.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5 mb-4">
                      <span className="text-2xs text-muted mr-1">Related expressions:</span>
                      {thought.related_expressions.map((expr, i) => (
                        <span key={i} className="text-2xs px-2 py-0.5 bg-ivory text-charcoal/80 rounded border border-border/70">
                          {expr}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Collapsible Evidence Section */}
                  <div className="border-t border-border/60 pt-3">
                    <button
                      onClick={() => toggleEvidence(evidenceId)}
                      className="inline-flex items-center gap-1.5 text-xs text-forest hover:text-forest-dark font-medium cursor-pointer transition-colors"
                    >
                      <span>{isExpanded ? 'Hide supporting evidence' : 'View supporting evidence'}</span>
                      {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    </button>

                    {isExpanded && (
                      <div className="mt-3.5 p-4 bg-ivory/70 rounded-lg border border-border/80 space-y-3 text-xs">
                        <div className="flex flex-wrap gap-x-4 gap-y-1 text-muted">
                          <span><strong>Timespan:</strong> {thought.date_range}</span>
                          <span><strong>Coverage:</strong> {thought.percentage}% of your reflections</span>
                        </div>

                        {renderEvidenceEntries(evidence.matching_entries, 'Reflections containing this phrase')}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* 4. CHANGES OVER TIME */}
      {changesOverTime.length > 0 && (
        <section className="space-y-4">
          <div className="border-b border-border pb-2 flex items-baseline justify-between">
            <div className="flex items-center gap-2.5">
              <Clock className="w-5 h-5 text-forest" strokeWidth={1.5} />
              <h2 className="font-serif text-xl md:text-2xl text-charcoal">Changes Over Time</h2>
            </div>
            <span className="text-xs text-muted">Chronological Comparison</span>
          </div>
          <p className="text-xs text-muted max-w-2xl">
            Observable differences between earlier and recent journal entries. These descriptions reflect temporal associations without implying causality.
          </p>

          <div className="space-y-4 mt-4">
            {changesOverTime.map((change, idx) => {
              const evidenceId = `change-${idx}`;
              const isExpanded = !!expandedEvidence[evidenceId];
              const evidence = change.evidence || {};

              return (
                <div
                  key={idx}
                  className="bg-surface border border-border rounded-xl p-6 shadow-2xs hover:border-forest/30 transition-colors"
                >
                  <h3 className="font-serif text-lg text-charcoal font-medium mb-1.5">{change.title}</h3>
                  <p className="text-sm text-charcoal/90 leading-relaxed mb-4">
                    {change.description}
                  </p>

                  {/* Comparison columns */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-ivory p-4 rounded-xl border border-border/80 text-xs mb-4">
                    <div className="space-y-1">
                      <span className="text-muted block text-2xs uppercase tracking-wider font-semibold">Earlier Phase</span>
                      <p className="text-sm font-serif font-medium text-charcoal">
                        {change.earlier_period?.primary_focus || 'Early Period'}
                      </p>
                      <p className="text-muted text-xs">{change.earlier_period?.date_range}</p>
                      {change.earlier_period?.excerpt && (
                        <p className="italic text-charcoal/80 border-l-2 border-sage/60 pl-2.5 mt-2">
                          &ldquo;{change.earlier_period.excerpt}&rdquo;
                        </p>
                      )}
                    </div>

                    <div className="space-y-1 md:border-l md:border-border md:pl-4">
                      <span className="text-muted block text-2xs uppercase tracking-wider font-semibold">Recent Phase</span>
                      <p className="text-sm font-serif font-medium text-charcoal">
                        {change.recent_period?.primary_focus || 'Recent Period'}
                      </p>
                      <p className="text-muted text-xs">{change.recent_period?.date_range}</p>
                      {change.recent_period?.excerpt && (
                        <p className="italic text-charcoal/80 border-l-2 border-forest/60 pl-2.5 mt-2">
                          &ldquo;{change.recent_period.excerpt}&rdquo;
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Collapsible Evidence Section */}
                  <div className="border-t border-border/60 pt-3">
                    <button
                      onClick={() => toggleEvidence(evidenceId)}
                      className="inline-flex items-center gap-1.5 text-xs text-forest hover:text-forest-dark font-medium cursor-pointer transition-colors"
                    >
                      <span>{isExpanded ? 'Hide supporting evidence' : 'View supporting evidence'}</span>
                      {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    </button>

                    {isExpanded && (
                      <div className="mt-3.5 p-4 bg-ivory/70 rounded-lg border border-border/80 space-y-3 text-xs">
                        {renderEvidenceEntries(evidence.earlier_matching, 'Earlier period representative entries')}
                        {renderEvidenceEntries(evidence.recent_matching, 'Recent period representative entries')}
                        {evidence.disclaimer && (
                          <p className="text-2xs text-muted/80 italic pt-1">{evidence.disclaimer}</p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* 5. PERSONAL OBSERVATIONS & REFLECTIONS */}
      {personalObservations.length > 0 && (
        <section className="space-y-4">
          <div className="border-b border-border pb-2 flex items-baseline justify-between">
            <div className="flex items-center gap-2.5">
              <Sparkles className="w-5 h-5 text-forest" strokeWidth={1.5} />
              <h2 className="font-serif text-xl md:text-2xl text-charcoal">Personal Observations</h2>
            </div>
            <span className="text-xs text-muted">Grounded Synthesis</span>
          </div>
          <p className="text-xs text-muted max-w-2xl">
            Thoughtful syntheses derived from your writing history, paired with questions for self-reflection. Non-diagnostic.
          </p>

          <div className="space-y-4 mt-4">
            {personalObservations.map((obs, idx) => {
              const evidenceId = `obs-${idx}`;
              const isExpanded = !!expandedEvidence[evidenceId];
              const evidence = obs.evidence || {};

              return (
                <div
                  key={idx}
                  className="bg-surface border border-border rounded-xl p-6 shadow-2xs hover:border-forest/30 transition-colors"
                >
                  <h3 className="font-serif text-lg text-charcoal font-medium mb-2">{obs.title}</h3>
                  <p className="text-sm text-charcoal/90 leading-relaxed mb-4">
                    {obs.summary}
                  </p>

                  {/* Reflective Question Box */}
                  {obs.reflective_question && (
                    <div className="bg-cream/70 border border-sage/30 rounded-xl p-4 my-4">
                      <div className="flex items-center gap-2 text-forest mb-1 text-xs font-semibold uppercase tracking-wider">
                        <BookOpen className="w-4 h-4" />
                        <span>Reflective Inquiry</span>
                      </div>
                      <p className="font-serif text-charcoal italic text-sm leading-relaxed">
                        &ldquo;{obs.reflective_question}&rdquo;
                      </p>
                    </div>
                  )}

                  {/* Collapsible Evidence Section */}
                  <div className="border-t border-border/60 pt-3">
                    <button
                      onClick={() => toggleEvidence(evidenceId)}
                      className="inline-flex items-center gap-1.5 text-xs text-forest hover:text-forest-dark font-medium cursor-pointer transition-colors"
                    >
                      <span>{isExpanded ? 'Hide supporting evidence' : 'View supporting evidence'}</span>
                      {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    </button>

                    {isExpanded && (
                      <div className="mt-3.5 p-4 bg-ivory/70 rounded-lg border border-border/80 space-y-2 text-xs text-muted">
                        <p><strong>Evidence basis:</strong> {evidence.basis || 'Derived from chronological entry progression.'}</p>
                        {evidence.observed_span && <p><strong>Observed span:</strong> {evidence.observed_span}</p>}
                        {renderEvidenceEntries(evidence.matching_entries, 'Supporting entries')}
                        <p className="text-2xs text-muted/80 italic pt-1">
                          Pensieve does not diagnose psychological conditions or infer medical certainty.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Backwards Compatibility / Fallback for Legacy Observation Strings if present without new structure */}
      {recurringTopics.length === 0 && emotionalTrends.length === 0 && patterns.emotion_trends?.observations?.length > 0 && (
        <section className="bg-surface border border-border rounded-xl p-6 shadow-2xs">
          <h2 className="font-serif text-lg text-charcoal mb-2">Observations</h2>
          <div className="space-y-2">
            {patterns.emotion_trends.observations.map((obs, i) => (
              <p key={i} className="text-sm text-muted leading-relaxed">• {obs}</p>
            ))}
          </div>
        </section>
      )}

      {/* Footer Nav */}
      <div className="border-t border-border pt-6 flex items-center justify-between text-xs text-muted">
        <Link to="/journal" className="inline-flex items-center gap-1.5 hover:text-charcoal transition-colors">
          <PenLine className="w-3.5 h-3.5" />
          <span>Write today&apos;s reflection</span>
        </Link>
        <Link to="/reflections" className="inline-flex items-center gap-1.5 text-forest hover:text-forest-dark font-medium transition-colors">
          <span>Explore Grounded Reflections</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </div>
    </div>
  );
}

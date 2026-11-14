import { useState, useEffect, useRef, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { reflectionsAPI, analysisAPI, conceptsAPI } from '../api/client';
import {
  Sparkles,
  Loader2,
  ChevronDown,
  ChevronUp,
  BookOpen,
  Compass,
  Calendar,
  ArrowRight,
  ExternalLink,
  ShieldCheck,
  AlertCircle,
  Quote,
  PenLine,
  History as HistoryIcon,
} from 'lucide-react';
import EmptyState from '../components/common/EmptyState';

function formatDate(dateStr) {
  if (!dateStr) return '';
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });
}

function dedupeReflections(items) {
  const seen = new Set();
  return (items || []).filter((item) => {
    if (!item || item.id == null) return false;
    if (seen.has(item.id)) return false;
    seen.add(item.id);
    return true;
  });
}

function extractThoughtToConsider(text) {
  if (!text) return null;
  // Match the final sentence ending with a question mark
  const match = text.match(/([A-Z][^.?!]*\?)\s*$/);
  if (match) return match[1].trim();

  // Fallback: check paragraphs in reverse for a sentence ending with '?'
  const paragraphs = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  for (let i = paragraphs.length - 1; i >= 0; i--) {
    const p = paragraphs[i];
    if (p.includes('?')) {
      const qMatch = p.match(/([^.?!]+\?)\s*$/);
      if (qMatch) return qMatch[1].trim();
      return p;
    }
  }
  return null;
}

export default function Reflections() {
  const [reflections, setReflections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [patterns, setPatterns] = useState(null);
  const [conceptCatalog, setConceptCatalog] = useState({});
  const [expandedConcepts, setExpandedConcepts] = useState({});
  const [showHistory, setShowHistory] = useState(false);
  const [expandedHistory, setExpandedHistory] = useState({});
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('info'); // 'info' | 'error' | 'warning'
  const [insufficientData, setInsufficientData] = useState(null);

  // Synchronous guard against rapid double clicks or concurrent requests
  const isSubmittingRef = useRef(false);

  // 1. Fetch user's reflection history
  const loadReflections = useCallback(async () => {
    try {
      const data = await reflectionsAPI.list();
      setReflections(dedupeReflections(data));
    } catch (err) {
      setMessage(err.message || 'Unable to load reflections history.');
      setMessageType('error');
    } finally {
      setLoading(false);
    }
  }, []);

  // 2. Fetch Phase 3 longitudinal pattern signals
  const loadPatterns = useCallback(async () => {
    try {
      const data = await analysisAPI.patterns();
      setPatterns(data);
      if (data?.status === 'insufficient_data') {
        setInsufficientData({
          entryCount: data.entry_count || 0,
          minRequired: data.safeguards?.min_entries || 3,
          spanDays: data.safeguards?.timespan_days || data.time_span_days || 0,
          minSpan: data.safeguards?.min_timespan_days || 7,
          message: data.message,
        });
      } else {
        setInsufficientData(null);
      }
    } catch {
      // Non-blocking for reflections page
    }
  }, []);

  // 3. Fetch Phase 4 concept catalog for expandable definitions
  const loadConcepts = useCallback(async () => {
    try {
      const docs = await conceptsAPI.list();
      if (Array.isArray(docs)) {
        const map = {};
        docs.forEach((doc) => {
          if (doc.id) map[doc.id] = doc;
        });
        setConceptCatalog(map);
      }
    } catch {
      // Non-blocking
    }
  }, []);

  useEffect(() => {
    loadReflections();
    loadPatterns();
    loadConcepts();
  }, [loadReflections, loadPatterns, loadConcepts]);

  // Handle generating a new reflection
  const handleGenerate = async () => {
    if (isSubmittingRef.current || generating) return;
    isSubmittingRef.current = true;
    setGenerating(true);
    setMessage('');

    try {
      const result = await reflectionsAPI.suggest();
      if (result.status === 'success' && result.reflection) {
        // Prepend new reflection and deduplicate strictly by ID
        setReflections((prev) => dedupeReflections([result.reflection, ...prev]));
        setMessage('');
        setInsufficientData(null);
        // Refresh longitudinal pattern analysis in background
        loadPatterns();
      } else if (result.status === 'policy_rejected') {
        const reason = result.reason;
        if (reason === 'insufficient_entries' || reason === 'insufficient_timespan') {
          setInsufficientData({
            entryCount: result.details?.entry_count || 0,
            minRequired: result.details?.min_required || 3,
            spanDays: result.details?.span_days || 0,
            minSpan: result.details?.min_required || 7,
            message: result.message || 'More journal entries are needed before generating a reflection.',
          });
        } else {
          setMessage(result.message || result.reason || 'More journal entries are needed before generating a reflection.');
          setMessageType('info');
        }
      } else {
        setMessage(result.message || 'Unable to generate reflection at this time.');
        setMessageType('info');
      }
    } catch (err) {
      setMessage(err.message || 'Failed to generate reflection.');
      setMessageType('error');
    } finally {
      setGenerating(false);
      isSubmittingRef.current = false;
    }
  };

  const toggleConceptExpand = (conceptId) => {
    setExpandedConcepts((prev) => ({
      ...prev,
      [conceptId]: !prev[conceptId],
    }));
  };

  const toggleHistoryItemExpand = (id) => {
    setExpandedHistory((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  // Separation of latest vs previous reflections
  // Latest reflection appears once prominently; previous reflections appear in history
  const uniqueReflections = dedupeReflections(reflections);
  const latestReflection = uniqueReflections.length > 0 ? uniqueReflections[0] : null;
  const previousReflections = uniqueReflections.slice(1);

  // Extract closing open-ended question for "A thought to consider"
  const thoughtToConsider = latestReflection ? extractThoughtToConsider(latestReflection.reflection_text) : null;

  // Extract Phase 3 observations for "What Pensieve noticed"
  const phase3Observations = (() => {
    const list = [];
    // Priority 1: Observations recorded directly in reflection input_summary
    if (latestReflection?.input_summary?.observations && Array.isArray(latestReflection.input_summary.observations)) {
      list.push(...latestReflection.input_summary.observations);
    }
    // Priority 2: Pattern synthesizer personalized observations
    if (list.length < 3 && patterns?.personal_observations && Array.isArray(patterns.personal_observations)) {
      patterns.personal_observations.forEach((obs) => {
        if (list.length < 3 && obs.summary && !list.includes(obs.summary)) {
          list.push(obs.summary);
        }
      });
    }
    // Priority 3: Emotion and theme trend observations
    if (list.length < 3 && patterns?.emotion_trends?.observations) {
      patterns.emotion_trends.observations.forEach((obs) => {
        if (list.length < 3 && !list.includes(obs)) {
          list.push(obs);
        }
      });
    }
    if (list.length < 3 && patterns?.theme_trends?.observations) {
      patterns.theme_trends.observations.forEach((obs) => {
        if (list.length < 3 && !list.includes(obs)) {
          list.push(obs);
        }
      });
    }
    return list.slice(0, 3);
  })();

  // Supporting evidence entries from longitudinal patterns
  const matchingEvidenceEntries = patterns?.personal_observations?.[0]?.evidence?.matching_entries || [];

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] py-16">
        <div className="w-10 h-10 rounded-full border-2 border-forest/20 border-t-forest animate-spin mb-4" />
        <div className="text-charcoal font-serif text-lg tracking-wide">
          Gathering your reflections...
        </div>
        <p className="text-xs text-muted mt-2">
          Retrieving journal trajectories and grounded psychological concepts.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto pb-16">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8 pb-6 border-b border-border/60">
        <div>
          <h1 className="font-serif text-3xl text-charcoal tracking-tight mb-1">Reflections</h1>
          <p className="text-sm text-muted">
            A personal, thoughtful reflection companion grounded in your journal.
          </p>
        </div>

        <button
          onClick={handleGenerate}
          disabled={generating}
          aria-label="Generate new reflection"
          className="inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-forest text-white rounded-lg hover:bg-forest-dark transition-all text-sm font-medium disabled:opacity-60 disabled:cursor-not-allowed shadow-xs cursor-pointer"
        >
          {generating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-white" />
              <span>Reflecting...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4 text-cream" />
              <span>Generate reflection</span>
            </>
          )}
        </button>
      </div>

      {/* Active Generating Banner */}
      {generating && (
        <div className="mb-8 p-6 bg-surface border border-forest/20 rounded-2xl shadow-2xs animate-pulse">
          <div className="flex items-center gap-3">
            <Loader2 className="w-5 h-5 text-forest animate-spin flex-shrink-0" />
            <div>
              <h3 className="font-serif text-base text-charcoal font-medium">Pensieve is reflecting...</h3>
              <p className="text-xs text-muted mt-0.5">
                Synthesizing observable writing patterns across your journal and anchoring them with established concepts.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Notice / Alert Banner */}
      {message && (
        <div
          role="alert"
          className={`flex items-start gap-3 text-sm rounded-xl px-5 py-4 mb-8 shadow-2xs ${
            messageType === 'error'
              ? 'text-red-700 bg-red-50/80 border border-red-200'
              : 'text-charcoal bg-surface border border-border'
          }`}
        >
          <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-current opacity-80" />
          <div className="flex-1">
            <p className="leading-relaxed">{message}</p>
          </div>
          {messageType === 'error' && (
            <button
              onClick={handleGenerate}
              className="text-xs font-medium underline hover:text-red-900 cursor-pointer flex-shrink-0 ml-2"
            >
              Try again
            </button>
          )}
        </div>
      )}

      {/* Insufficient Data State */}
      {insufficientData && uniqueReflections.length === 0 && (
        <div className="bg-surface border border-border rounded-2xl p-8 sm:p-12 text-center my-6 shadow-2xs max-w-2xl mx-auto">
          <div className="w-12 h-12 rounded-full bg-forest/10 border border-forest/20 flex items-center justify-center mx-auto mb-4 text-forest">
            <PenLine className="w-6 h-6" strokeWidth={1.75} />
          </div>
          <h2 className="font-serif text-2xl text-charcoal mb-2">More writing needed for reflections</h2>
          <p className="text-sm text-muted max-w-md mx-auto mb-6 leading-relaxed">
            {insufficientData.message ||
              `Pensieve requires at least ${insufficientData.minRequired} journal entries across ${insufficientData.minSpan} or more days to observe genuine trajectories. This ensures reflections are grounded in sustained patterns rather than fleeting moments.`}
          </p>

          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-ivory border border-border text-xs text-muted mb-8">
            <Calendar className="w-3.5 h-3.5 text-forest" />
            <span>
              Current: <strong>{insufficientData.entryCount}</strong> of {insufficientData.minRequired} entries
              {insufficientData.spanDays > 0 ? ` (${insufficientData.spanDays} days)` : ''}
            </span>
          </div>

          <div>
            <Link
              to="/journal"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-forest text-white rounded-lg hover:bg-forest-dark transition-colors text-sm font-medium shadow-xs"
            >
              <PenLine className="w-4 h-4" />
              <span>Write a journal entry</span>
            </Link>
          </div>
        </div>
      )}

      {/* Empty State (when user has no reflections and not in insufficient state) */}
      {!insufficientData && uniqueReflections.length === 0 && !generating && (
        <div className="py-8">
          <EmptyState
            icon={Sparkles}
            title="No reflections yet"
            description="Pensieve is ready to synthesize your journal trajectories into a grounded reflection with psychological and philosophical frameworks."
            actionLabel="Generate your first reflection"
            onAction={handleGenerate}
          />
        </div>
      )}

      {/* Main Content Hierarchy (When reflections exist) */}
      {latestReflection && (
        <div className="space-y-10">
          {/* SECTION A: LATEST REFLECTION */}
          <article className="bg-surface border border-border/80 rounded-2xl p-6 sm:p-9 shadow-xs transition-shadow">
            {/* Header Badge & Date */}
            <div className="flex flex-wrap items-center justify-between gap-3 mb-6 pb-4 border-b border-border/50">
              <div className="flex items-center gap-2.5">
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-forest/10 text-forest border border-forest/15">
                  <Sparkles className="w-3 h-3 text-forest" />
                  Latest Reflection
                </span>
                <span className="text-xs text-muted">
                  {formatDate(latestReflection.created_at)}
                </span>
              </div>

              {latestReflection.confidence != null && (
                <div className="flex items-center gap-1.5 text-xs text-muted" title="Reflects textual observation confidence">
                  <ShieldCheck className="w-3.5 h-3.5 text-forest" />
                  <span>{(latestReflection.confidence * 100).toFixed(0)}% confidence</span>
                </div>
              )}
            </div>

            {/* Reflection Content (Personal, readable, 2-3 paragraphs) */}
            <div className="space-y-4 mb-8">
              {latestReflection.reflection_text
                .split(/\n\s*\n/)
                .filter(Boolean)
                .map((paragraph, pIdx) => (
                  <p
                    key={pIdx}
                    className="font-serif text-base sm:text-lg text-charcoal leading-relaxed tracking-normal font-normal selection:bg-cream"
                  >
                    {paragraph}
                  </p>
                ))}
            </div>

            {/* Mandatory Disclaimer Footnote */}
            <div className="pt-4 border-t border-border/40">
              <p className="text-[11px] text-muted/70 leading-normal italic">
                {latestReflection.disclaimer ||
                  'This reflection describes observable patterns in journal text for personal contemplation. It does not constitute psychological, psychiatric, or medical advice or diagnosis.'}
              </p>
            </div>
          </article>

          {/* SECTION C: A THOUGHT TO CONSIDER (Emotional Focal Point) */}
          {thoughtToConsider && (
            <section
              aria-label="A thought to consider"
              className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-surface via-surface to-ivory border-2 border-forest/20 p-8 sm:p-10 shadow-sm"
            >
              {/* Decorative background watermark */}
              <Quote
                className="absolute right-4 bottom-2 w-32 h-32 text-forest/[0.04] pointer-events-none select-none"
                strokeWidth={1}
              />

              <div className="relative z-10">
                <div className="flex items-center gap-2 text-forest text-xs font-semibold uppercase tracking-wider mb-4">
                  <Quote className="w-4 h-4 text-forest" />
                  <span>A thought to consider</span>
                </div>

                <blockquote className="font-serif text-xl sm:text-2xl text-charcoal italic leading-relaxed mb-4">
                  &ldquo;{thoughtToConsider}&rdquo;
                </blockquote>

                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pt-4 border-t border-forest/15 mt-6">
                  <p className="text-xs text-muted">
                    Take a quiet moment to sit with this question, or explore it in your next journal entry.
                  </p>
                  <Link
                    to="/journal"
                    className="inline-flex items-center gap-1.5 text-xs font-medium text-forest hover:text-forest-dark transition-colors flex-shrink-0 cursor-pointer"
                  >
                    <span>Write in journal</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </Link>
                </div>
              </div>
            </section>
          )}

          {/* SECTION B: WHAT PENSIEVE NOTICED */}
          {phase3Observations.length > 0 && (
            <section aria-label="What Pensieve noticed" className="bg-surface border border-border/80 rounded-2xl p-6 sm:p-8 shadow-xs">
              <div className="flex items-center gap-2 mb-2">
                <Compass className="w-5 h-5 text-forest" />
                <h2 className="font-serif text-xl text-charcoal font-medium">What Pensieve noticed</h2>
              </div>
              <p className="text-xs text-muted mb-6">
                Observable patterns detected across your journal entries over this period.
              </p>

              <div className="space-y-4">
                {phase3Observations.map((observation, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-3 p-4 rounded-xl bg-ivory/60 border border-border/60"
                  >
                    <div className="w-2 h-2 rounded-full bg-forest mt-2 flex-shrink-0" />
                    <div className="flex-1 text-sm text-charcoal leading-relaxed">
                      {observation}
                    </div>
                  </div>
                ))}
              </div>

              {/* Supporting Evidence Entries with Links */}
              {matchingEvidenceEntries.length > 0 && (
                <div className="mt-6 pt-5 border-t border-border/50">
                  <h4 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">
                    Supporting entries from your journal
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {matchingEvidenceEntries.map((entry) => (
                      <Link
                        key={entry.id}
                        to={`/journal?id=${entry.id}`}
                        className="group flex flex-col p-3 rounded-lg bg-surface border border-border hover:border-forest/40 hover:bg-ivory/40 transition-all text-left"
                      >
                        <div className="flex items-center justify-between text-xs text-muted mb-1">
                          <span className="font-medium text-charcoal group-hover:text-forest transition-colors">
                            {entry.title || 'Journal Entry'}
                          </span>
                          <span className="text-[11px]">{entry.date_short || entry.date}</span>
                        </div>
                        {entry.excerpt && (
                          <p className="text-xs text-muted line-clamp-2 italic font-serif">
                            &ldquo;{entry.excerpt}&rdquo;
                          </p>
                        )}
                        <span className="inline-flex items-center gap-1 text-[11px] text-forest font-medium mt-2 opacity-80 group-hover:opacity-100">
                          View entry <ArrowRight className="w-3 h-3" />
                        </span>
                      </Link>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}

          {/* SECTION D: GROUNDED IN */}
          {latestReflection.grounded_concepts && latestReflection.grounded_concepts.length > 0 && (
            <section aria-label="Grounded concepts" className="bg-surface border border-border/80 rounded-2xl p-6 sm:p-8 shadow-xs">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <BookOpen className="w-5 h-5 text-forest" />
                  <h2 className="font-serif text-xl text-charcoal font-medium">Grounded in</h2>
                </div>
                <Link
                  to="/concepts"
                  className="text-xs font-medium text-forest hover:text-forest-dark transition-colors inline-flex items-center gap-1"
                >
                  <span>Explore Concepts Library</span>
                  <ExternalLink className="w-3 h-3" />
                </Link>
              </div>
              <p className="text-xs text-muted mb-6">
                Established psychological and philosophical frameworks informing this reflection.
              </p>

              <div className="space-y-4">
                {latestReflection.grounded_concepts.map((concept, i) => {
                  const conceptId = typeof concept === 'string' ? concept : concept.concept_id || concept.id || `c-${i}`;
                  const conceptName = typeof concept === 'string' ? concept : concept.name || concept.concept || 'Concept';
                  const conceptSource = typeof concept === 'object' ? concept.source : '';
                  const fullDoc = conceptCatalog[conceptId] || {};
                  const isExpanded = !!expandedConcepts[conceptId];

                  return (
                    <div
                      key={conceptId || i}
                      className="border border-border rounded-xl p-4 bg-ivory/30 transition-all hover:border-forest/30"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1">
                          <div className="flex flex-wrap items-center gap-2 mb-1">
                            <span className="font-serif text-base font-medium text-forest">
                              {conceptName}
                            </span>
                            {fullDoc.category && (
                              <span className="text-[10px] font-medium uppercase tracking-wider px-2 py-0.5 rounded-full bg-forest/10 text-forest">
                                {fullDoc.category}
                              </span>
                            )}
                            {typeof concept === 'object' && concept.relevance_score != null && (
                              <span className="text-xs text-muted">
                                ({(concept.relevance_score * 100).toFixed(0)}% relevant)
                              </span>
                            )}
                          </div>

                          {conceptSource && (
                            <p className="text-xs text-muted">
                              <span className="font-medium text-charcoal/70">Source:</span> {conceptSource}
                            </p>
                          )}
                        </div>

                        <button
                          onClick={() => toggleConceptExpand(conceptId)}
                          aria-label={isExpanded ? 'Collapse concept details' : 'Expand concept details'}
                          className="inline-flex items-center gap-1 text-xs text-forest hover:text-forest-dark font-medium py-1 px-2 rounded-md hover:bg-forest/5 transition-colors cursor-pointer flex-shrink-0"
                        >
                          <span>{isExpanded ? 'Hide' : 'Details'}</span>
                          {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                        </button>
                      </div>

                      {/* Expandable Concept Detail */}
                      {isExpanded && (
                        <div className="mt-4 pt-3 border-t border-border/60 text-xs text-charcoal space-y-3 animate-fadeIn">
                          {fullDoc.definition && (
                            <div>
                              <strong className="text-charcoal/80 block mb-0.5">Definition:</strong>
                              <p className="text-muted leading-relaxed">{fullDoc.definition}</p>
                            </div>
                          )}

                          {fullDoc.explanation && (
                            <div>
                              <strong className="text-charcoal/80 block mb-0.5">Reflective Lens:</strong>
                              <p className="text-muted leading-relaxed">{fullDoc.explanation}</p>
                            </div>
                          )}

                          {fullDoc.cautions && fullDoc.cautions.length > 0 && (
                            <div className="p-2.5 rounded-lg bg-surface border border-border/50 text-muted">
                              <strong className="text-charcoal/80 block mb-0.5">Ethical Cautions:</strong>
                              <ul className="list-disc list-inside space-y-0.5 text-[11px]">
                                {fullDoc.cautions.map((caution, cIdx) => (
                                  <li key={cIdx}>{caution}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* SECTION E: PREVIOUS REFLECTIONS (Collapsible History) */}
          {previousReflections.length > 0 && (
            <section aria-label="Previous reflections" className="pt-4 border-t border-border/60">
              <button
                onClick={() => setShowHistory((prev) => !prev)}
                className="w-full flex items-center justify-between p-4 rounded-xl bg-surface border border-border hover:border-forest/40 transition-colors text-left cursor-pointer group shadow-2xs"
              >
                <div className="flex items-center gap-2.5">
                  <HistoryIcon className="w-4 h-4 text-forest" />
                  <span className="font-serif text-lg text-charcoal font-medium">
                    Previous reflections
                  </span>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-forest/10 text-forest font-medium">
                    {previousReflections.length}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-xs text-muted group-hover:text-forest transition-colors">
                  <span>{showHistory ? 'Collapse history' : 'View history'}</span>
                  {showHistory ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </div>
              </button>

              {showHistory && (
                <div className="mt-6 space-y-6 animate-fadeIn">
                  {previousReflections.map((reflection) => {
                    const isItemExpanded = !!expandedHistory[reflection.id];
                    const paragraphs = reflection.reflection_text
                      .split(/\n\s*\n/)
                      .filter(Boolean);

                    return (
                      <article
                        key={reflection.id}
                        className="bg-surface border border-border/70 rounded-xl p-6 shadow-2xs transition-all"
                      >
                        <div className="flex items-center justify-between gap-3 mb-4 pb-3 border-b border-border/40">
                          <span className="text-xs font-medium text-muted">
                            {formatDate(reflection.created_at)}
                          </span>
                          {reflection.confidence != null && (
                            <span className="text-xs text-muted">
                              {(reflection.confidence * 100).toFixed(0)}% confidence
                            </span>
                          )}
                        </div>

                        {/* Reflection content preview or full */}
                        <div className="space-y-3 mb-4">
                          {isItemExpanded ? (
                            paragraphs.map((p, pIdx) => (
                              <p key={pIdx} className="font-serif text-sm sm:text-base text-charcoal leading-relaxed">
                                {p}
                              </p>
                            ))
                          ) : (
                            <p className="font-serif text-sm sm:text-base text-charcoal leading-relaxed line-clamp-3">
                              {paragraphs[0] || reflection.reflection_text}
                            </p>
                          )}
                        </div>

                        {paragraphs.length > 1 && (
                          <button
                            onClick={() => toggleHistoryItemExpand(reflection.id)}
                            className="text-xs font-medium text-forest hover:text-forest-dark transition-colors mb-4 cursor-pointer inline-flex items-center gap-1"
                          >
                            <span>{isItemExpanded ? 'Show less' : 'Read full reflection'}</span>
                            {isItemExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                          </button>
                        )}

                        {/* Grounded concepts pills */}
                        {reflection.grounded_concepts && reflection.grounded_concepts.length > 0 && (
                          <div className="flex flex-wrap items-center gap-1.5 pt-3 border-t border-border/40">
                            <span className="text-[11px] text-muted mr-1">Grounded in:</span>
                            {reflection.grounded_concepts.map((concept, cIdx) => (
                              <span
                                key={cIdx}
                                className="text-[11px] font-medium px-2 py-0.5 rounded-md bg-ivory border border-border text-charcoal/80"
                              >
                                {typeof concept === 'string' ? concept : concept.name || concept.concept}
                              </span>
                            ))}
                          </div>
                        )}
                      </article>
                    );
                  })}
                </div>
              )}
            </section>
          )}
        </div>
      )}
    </div>
  );
}

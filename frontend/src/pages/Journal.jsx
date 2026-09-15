import { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { entriesAPI, analysisAPI } from '../api/client';
import {
  Save,
  Sparkles,
  Eye,
  EyeOff,
  Check,
  Loader2,
  TrendingUp,
  Lightbulb,
  ArrowRight,
  Smile,
  Lock,
} from 'lucide-react';

export default function Journal() {
  const [searchParams] = useSearchParams();
  const editId = searchParams.get('id');

  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [mood, setMood] = useState('');
  const [entryId, setEntryId] = useState(editId ? parseInt(editId, 10) : null);
  const [isDraft, setIsDraft] = useState(true);
  const [saveStatus, setSaveStatus] = useState('new'); // 'new', 'saving', 'saved', 'unsaved'
  const [analysis, setAnalysis] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [focusMode, setFocusMode] = useState(false);
  const [loading, setLoading] = useState(!!editId);
  const [error, setError] = useState('');

  const autosaveTimer = useRef(null);
  const contentRef = useRef(null);

  // Load existing entry if ?id= is present, or reset for new entry
  useEffect(() => {
    if (editId) {
      setLoading(true);
      entriesAPI
        .get(editId)
        .then((entry) => {
          setTitle(entry.title || '');
          setContent(entry.content || '');
          setMood(entry.mood || '');
          setEntryId(entry.id);
          setIsDraft(entry.is_draft);
          setAnalysis(entry.analysis || null);
          setSaveStatus('saved');
          setLoading(false);
        })
        .catch(() => {
          setError('Could not load the requested journal entry.');
          setLoading(false);
        });
    } else {
      setTitle('');
      setContent('');
      setMood('');
      setEntryId(null);
      setIsDraft(true);
      setAnalysis(null);
      setSaveStatus('new');
      setError('');
      setLoading(false);
    }
  }, [editId]);

  // Debounced autosave implementation
  const performAutosave = useCallback(
    async (currentTitle, currentContent, currentMood, currentEntryId) => {
      if (!currentContent.trim()) return;
      setSaveStatus('saving');
      try {
        const payload = {
          title: currentTitle || 'Untitled',
          content: currentContent,
          mood: currentMood || undefined,
        };
        if (currentEntryId) payload.entry_id = currentEntryId;
        const result = await entriesAPI.autosave(payload);
        if (!currentEntryId && result.id) {
          setEntryId(result.id);
        }
        setSaveStatus('saved');
      } catch {
        setSaveStatus('unsaved');
      }
    },
    [],
  );

  const scheduleAutosave = useCallback(
    (newTitle, newContent, newMood, currentEntryId) => {
      setSaveStatus('unsaved');
      if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
      autosaveTimer.current = setTimeout(() => {
        performAutosave(newTitle, newContent, newMood, currentEntryId);
      }, 2000);
    },
    [performAutosave],
  );

  const handleTitleChange = (e) => {
    const v = e.target.value;
    setTitle(v);
    scheduleAutosave(v, content, mood, entryId);
  };

  const handleContentChange = (e) => {
    const v = e.target.value;
    setContent(v);
    scheduleAutosave(title, v, mood, entryId);
  };

  const handleMoodChange = (e) => {
    const v = e.target.value;
    setMood(v);
    scheduleAutosave(title, content, v, entryId);
  };

  // Finalize / Manual Save
  const handleSave = async () => {
    if (!content.trim()) return;
    if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    setSaveStatus('saving');
    setError('');
    try {
      const payload = {
        title: title.trim() || 'Untitled reflection',
        content,
        mood: mood || undefined,
        is_draft: false,
      };
      let result;
      if (entryId) {
        result = await entriesAPI.update(entryId, payload);
      } else {
        result = await entriesAPI.create(payload);
        setEntryId(result.id);
      }
      setIsDraft(false);
      setSaveStatus('saved');
    } catch (err) {
      setError(err.message || 'Failed to save entry.');
      setSaveStatus('unsaved');
    }
  };

  // Real ML Analysis Execution
  const handleAnalyze = async () => {
    if (!content.trim()) return;
    if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    setAnalyzing(true);
    setError('');

    let currentId = entryId;

    // Always persist & publish as non-draft before running analysis
    try {
      const payload = {
        title: title.trim() || 'Untitled reflection',
        content,
        mood: mood || undefined,
        is_draft: false,
      };
      if (currentId) {
        await entriesAPI.update(currentId, payload);
      } else {
        const created = await entriesAPI.create(payload);
        currentId = created.id;
        setEntryId(created.id);
      }
      setIsDraft(false);
      setSaveStatus('saved');
    } catch (err) {
      setError(err.message || 'Failed to save entry prior to analysis.');
      setAnalyzing(false);
      return;
    }

    try {
      const result = await analysisAPI.analyze(currentId);
      setAnalysis(result);
    } catch (err) {
      setError(err.message || 'Analysis failed.');
    } finally {
      setAnalyzing(false);
    }
  };

  const wordCount = content.trim() ? content.trim().split(/\s+/).filter(Boolean).length : 0;
  const todayFormatted = new Intl.DateTimeFormat('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  }).format(new Date());

  const topEmotionName =
    typeof analysis?.top_emotion === 'object' && analysis.top_emotion !== null
      ? analysis.top_emotion.emotion
      : analysis?.top_emotion;

  const topEmotionScore =
    typeof analysis?.top_emotion === 'object' && analysis.top_emotion?.score != null
      ? `${Math.round(analysis.top_emotion.score * 100)}%`
      : null;

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[460px]">
        <div className="text-muted/80 font-serif text-lg tracking-wide animate-pulse">
          Opening your notebook...
        </div>
      </div>
    );
  }

  return (
    <div className={`transition-all duration-300 ${focusMode ? 'py-4 px-2' : 'space-y-10'}`}>
      {/* 1. TOP REFINED JOURNAL HEADER */}
      {!focusMode && (
        <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-border/80">
          {/* Header Left: Current date & subtle contextual label */}
          <div>
            <h2 className="font-serif text-base sm:text-lg font-medium text-charcoal tracking-tight">
              {todayFormatted}
            </h2>
            <p className="text-xs text-muted mt-0.5 tracking-wide">
              {editId ? `Editing reflection #${editId}` : 'Daily reflection'}
            </p>
          </div>

          {/* Header Right: Autosave status, word count, focus mode, and action buttons */}
          <div className="flex items-center flex-wrap gap-2.5 sm:gap-3.5">
            {/* Autosave Status */}
            <div className="flex items-center gap-1.5 text-xs text-muted pr-1 select-none">
              {saveStatus === 'saving' && (
                <Loader2 className="w-3.5 h-3.5 animate-spin text-forest" />
              )}
              {saveStatus === 'saved' && (
                <Check className="w-3.5 h-3.5 text-forest" strokeWidth={2.2} />
              )}
              {saveStatus === 'unsaved' && (
                <span className="w-1.5 h-1.5 rounded-full bg-muted/60" />
              )}
              {saveStatus === 'new' && (
                <span className="w-1.5 h-1.5 rounded-full bg-border" />
              )}
              <span className="capitalize">
                {saveStatus === 'new'
                  ? 'New draft'
                  : saveStatus === 'saving'
                  ? 'Saving...'
                  : saveStatus === 'saved'
                  ? 'Saved'
                  : 'Unsaved edits'}
              </span>
            </div>

            {/* Word Count */}
            <div className="inline-flex items-center px-2.5 py-1 rounded-md bg-surface border border-border text-xs text-muted font-sans select-none">
              {wordCount} {wordCount === 1 ? 'word' : 'words'}
            </div>

            {/* Focus Mode Toggle */}
            <button
              onClick={() => setFocusMode(true)}
              className="p-1.5 text-muted hover:text-charcoal hover:bg-surface rounded-lg border border-transparent hover:border-border transition-colors cursor-pointer"
              title="Enter focus mode"
              aria-label="Enter focus mode"
            >
              <Eye className="w-4 h-4" strokeWidth={1.8} />
            </button>

            {/* Save Button */}
            <button
              onClick={handleSave}
              disabled={!content.trim() || saveStatus === 'saving'}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-surface hover:bg-ivory border border-border text-charcoal rounded-lg transition-colors text-xs font-medium disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer shadow-2xs"
            >
              <Save className="w-3.5 h-3.5 text-muted" />
              <span>Save</span>
            </button>

            {/* Analyze Entry Button - Prominent yet elegant */}
            <button
              onClick={handleAnalyze}
              disabled={analyzing || !content.trim()}
              className="inline-flex items-center gap-2 px-3.5 py-1.5 bg-forest hover:bg-forest-dark text-white rounded-lg transition-all duration-150 text-xs font-medium shadow-xs disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            >
              {analyzing ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Sparkles className="w-3.5 h-3.5 text-cream" />
              )}
              <span>{analyzing ? 'Analyzing entry...' : 'Analyze Entry'}</span>
            </button>
          </div>
        </header>
      )}

      {/* Floating Exit Button for Focus Mode */}
      {focusMode && (
        <div className="fixed top-6 right-6 z-50">
          <button
            onClick={() => setFocusMode(false)}
            className="inline-flex items-center gap-2 px-4 py-2 bg-surface border border-border shadow-md rounded-full text-xs font-medium text-charcoal hover:bg-cream transition-colors cursor-pointer"
          >
            <EyeOff className="w-3.5 h-3.5" />
            <span>Exit focus mode</span>
          </button>
        </div>
      )}

      {/* Error Notice */}
      {error && (
        <div className="max-w-[760px] mx-auto text-xs text-red-800 bg-red-50/90 border border-red-200 rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      {/* 2. CENTERED EDITORIAL WRITING CANVAS */}
      <div className="max-w-[760px] mx-auto bg-surface border border-border/80 rounded-2xl shadow-[0_1px_3px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.02)] px-7 sm:px-14 py-8 sm:py-12 transition-all">
        {/* Subtle Folio Header / Mood & Status Bar */}
        {!focusMode && (
          <div className="flex items-center justify-between pb-4 border-b border-border/60 text-xs text-muted mb-6 select-none">
            {/* Mood input integrated as an editorial prompt */}
            <div className="flex items-center gap-2 flex-1 mr-4">
              <Smile className="w-3.5 h-3.5 text-muted/70 flex-shrink-0" />
              <input
                type="text"
                value={mood}
                onChange={handleMoodChange}
                placeholder="How does today feel? (e.g. calm, uncertain, focused)"
                className="w-full bg-transparent border-none text-xs text-charcoal placeholder:text-muted/50 focus:outline-none truncate"
              />
            </div>

            {/* Draft / Saved Status */}
            <div className="flex items-center gap-2 flex-shrink-0">
              {isDraft ? (
                <span className="text-[10px] uppercase font-semibold tracking-wider text-muted/80 bg-cream/70 px-2 py-0.5 rounded border border-border/50">
                  Draft
                </span>
              ) : (
                <span className="text-[10px] uppercase font-semibold tracking-wider text-forest bg-forest/10 px-2 py-0.5 rounded border border-forest/20">
                  Saved
                </span>
              )}
            </div>
          </div>
        )}

        {/* Title Input */}
        <div className="mb-4">
          <input
            type="text"
            value={title}
            onChange={handleTitleChange}
            placeholder="Title your reflection..."
            className="w-full font-serif text-3xl sm:text-4xl text-charcoal placeholder:text-muted/40 bg-transparent border-none focus:outline-none font-normal leading-tight tracking-tight"
          />
        </div>

        {/* Journal Body Textarea with intentional Empty State */}
        <div
          className="relative min-h-[500px] cursor-text"
          onClick={() => contentRef.current?.focus()}
        >
          <textarea
            ref={contentRef}
            value={content}
            onChange={handleContentChange}
            placeholder={content.trim() ? '' : 'Begin writing...'}
            className="w-full min-h-[500px] font-serif text-[18px] sm:text-[20px] text-charcoal leading-[1.85] bg-transparent border-none resize-none focus:outline-none placeholder:text-muted/40"
          />

          {/* Empty State Secondary Prompt */}
          {!content.trim() && (
            <div className="pointer-events-none absolute left-0 top-10 text-muted/55 font-serif text-sm italic select-none">
              Put down whatever feels present today.
            </div>
          )}
        </div>
      </div>

      {/* 3. ANALYSIS PREVIEW ("Your reflection signals") */}
      {analysis && !focusMode && (
        <section className="max-w-[760px] mx-auto pt-2 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-border/70">
            <div>
              <h3 className="font-serif text-xl sm:text-2xl text-charcoal font-medium tracking-tight">
                Your reflection signals
              </h3>
              <p className="text-xs text-muted mt-0.5">
                Quietly extracted from this entry by the ML pipeline
              </p>
            </div>
            <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-forest bg-forest/10 border border-forest/20 px-2.5 py-1 rounded-md">
              <Check className="w-3 h-3 text-forest" />
              <span>Analyzed</span>
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Card 1: Emotions */}
            <div className="bg-surface border border-border rounded-xl p-5 flex flex-col justify-between shadow-2xs">
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-muted mb-2.5">
                  Emotions
                </p>
                {topEmotionName && (
                  <div className="mb-3.5">
                    <span className="text-xs text-muted">Primary: </span>
                    <span className="text-xs font-semibold text-forest capitalize ml-1">
                      {topEmotionName}
                    </span>
                    {topEmotionScore && (
                      <span className="text-[11px] text-muted/80 ml-1.5">
                        ({topEmotionScore})
                      </span>
                    )}
                  </div>
                )}
                {analysis.emotions && (
                  <div className="space-y-2">
                    {Object.entries(analysis.emotions)
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 4)
                      .map(([emotion, score]) => (
                        <div key={emotion} className="flex items-center justify-between text-xs">
                          <span className="text-muted capitalize truncate w-24 text-[11px]">
                            {emotion}
                          </span>
                          <div className="flex-1 mx-2 bg-cream rounded-full h-1.5 overflow-hidden">
                            <div
                              className="bg-forest/80 rounded-full h-1.5 transition-all"
                              style={{ width: `${Math.min(score * 100, 100)}%` }}
                            />
                          </div>
                          <span className="text-muted text-[10px] w-7 text-right font-medium">
                            {(score * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                  </div>
                )}
              </div>
            </div>

            {/* Card 2: Semantic Theme */}
            <div className="bg-surface border border-border rounded-xl p-5 flex flex-col justify-between shadow-2xs">
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-muted mb-2.5">
                  Detected Theme
                </p>
                <p className="font-serif text-base text-charcoal font-medium leading-snug mb-3">
                  {analysis.theme || 'Reflective Journaling'}
                </p>
                {analysis.theme_cluster_id !== null && analysis.theme_cluster_id !== undefined && (
                  <span className="inline-block text-[10px] uppercase font-semibold tracking-wider text-forest bg-cream/90 border border-border/60 px-2 py-0.5 rounded">
                    Cluster {analysis.theme_cluster_id}
                  </span>
                )}
              </div>
              <p className="text-[11px] text-muted/70 mt-3 pt-3 border-t border-border/40">
                Aligned using Sentence-BERT embeddings
              </p>
            </div>

            {/* Card 3: Linguistic Observations */}
            <div className="bg-surface border border-border rounded-xl p-5 flex flex-col justify-between shadow-2xs">
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-muted mb-2.5">
                  Writing Observations
                </p>
                {analysis.linguistic_features ? (
                  <div className="grid grid-cols-2 gap-y-2.5 gap-x-2 text-xs">
                    <div>
                      <span className="text-muted block text-[10px]">Words</span>
                      <span className="text-charcoal font-medium">
                        {analysis.linguistic_features.word_count ?? wordCount}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted block text-[10px]">Sentences</span>
                      <span className="text-charcoal font-medium">
                        {analysis.linguistic_features.sentence_count ?? 1}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted block text-[10px]">Vocab Diversity</span>
                      <span className="text-charcoal font-medium">
                        {analysis.linguistic_features.vocabulary_diversity != null
                          ? `${(analysis.linguistic_features.vocabulary_diversity * 100).toFixed(0)}%`
                          : '—'}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted block text-[10px]">1st Person</span>
                      <span className="text-charcoal font-medium">
                        {analysis.linguistic_features.first_person_pronoun_count != null
                          ? `${analysis.linguistic_features.first_person_pronoun_count}`
                          : '—'}
                      </span>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-muted italic">Profile generated after save.</p>
                )}
              </div>
              <p className="text-[11px] text-muted/70 mt-3 pt-3 border-t border-border/40">
                Synthesized via spaCy NLP pipeline
              </p>
            </div>
          </div>

          {/* Navigation Action Links */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2 bg-surface border border-border/80 rounded-xl px-5 py-3 text-xs shadow-2xs">
            <Link
              to="/patterns"
              className="inline-flex items-center gap-1.5 text-forest hover:text-forest-dark font-medium transition-colors"
            >
              <TrendingUp className="w-3.5 h-3.5" />
              <span>View patterns</span>
              <ArrowRight className="w-3 h-3 ml-0.5" />
            </Link>

            <Link
              to="/reflections"
              className="inline-flex items-center gap-1.5 text-forest hover:text-forest-dark font-medium transition-colors"
            >
              <Lightbulb className="w-3.5 h-3.5" />
              <span>Generate reflection</span>
              <ArrowRight className="w-3 h-3 ml-0.5" />
            </Link>
          </div>
        </section>
      )}

      {/* 4. PRIVACY FOOTER */}
      {!focusMode && (
        <footer className="text-center pt-6 pb-6">
          <div className="inline-flex items-center gap-1.5 text-xs text-muted/70">
            <Lock className="w-3 h-3 text-muted/60" />
            <span>Your journal stays private to your account.</span>
          </div>
        </footer>
      )}
    </div>
  );
}

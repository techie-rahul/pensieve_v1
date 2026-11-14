import { useState, useEffect } from 'react';
import { reflectionsAPI } from '../api/client';
import { Lightbulb, Loader2, RefreshCw } from 'lucide-react';
import EmptyState from '../components/common/EmptyState';

function formatDate(dateStr) {
  return new Date(dateStr).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
}

export default function Reflections() {
  const [reflections, setReflections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('info'); // 'info' or 'error'

  useEffect(() => {
    reflectionsAPI.list()
      .then(setReflections)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleGenerate = async () => {
    setGenerating(true);
    setMessage('');
    try {
      const result = await reflectionsAPI.suggest();
      if (result.status === 'success' && result.reflection) {
        setReflections(prev => [result.reflection, ...prev]);
        setMessage('');
      } else if (result.status === 'policy_rejected') {
        setMessage(result.message || result.reason || 'More journal entries are needed before generating a reflection.');
        setMessageType('info');
      } else {
        setMessage(result.message || 'Unable to generate reflection at this time.');
        setMessageType('info');
      }
    } catch (err) {
      setMessage(err.message || 'Failed to generate reflection.');
      setMessageType('error');
    } finally {
      setGenerating(false);
    }
  };

  if (loading) {
    return <div className="text-muted py-12 text-center font-serif">Loading reflections...</div>;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-serif text-2xl text-charcoal mb-1">Reflections</h1>
          <p className="text-sm text-muted">Grounded observations drawn from your journal patterns.</p>
        </div>
        <button
          onClick={handleGenerate}
          disabled={generating}
          className="inline-flex items-center gap-2 px-4 py-2 bg-forest text-white rounded-lg hover:bg-forest-dark transition-colors text-sm font-medium disabled:opacity-50"
        >
          {generating ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
          {generating ? 'Generating...' : 'Generate reflection'}
        </button>
      </div>

      {message && (
        <div className={`text-sm rounded-lg px-4 py-3 mb-6 ${
          messageType === 'error'
            ? 'text-red-600 bg-red-50 border border-red-200'
            : 'text-muted bg-surface border border-border'
        }`}>
          {message}
        </div>
      )}

      {reflections.length === 0 ? (
        <EmptyState
          icon={Lightbulb}
          title="No reflections yet"
          description="Generate your first reflection after writing and analyzing a few journal entries."
        />
      ) : (
        <div className="space-y-8">
          {reflections.map((reflection, idx) => (
            <article key={reflection.id || idx} className="bg-surface border border-border rounded-xl px-6 py-6">
              {/* Reflection text */}
              <div className="mb-5">
                <p className="font-serif text-base text-charcoal leading-relaxed whitespace-pre-wrap">
                  {reflection.reflection_text}
                </p>
              </div>

              {/* Grounded concepts */}
              {reflection.grounded_concepts && reflection.grounded_concepts.length > 0 && (
                <div className="mb-4 pt-4 border-t border-border">
                  <h4 className="text-xs font-medium text-muted uppercase tracking-wide mb-3">Grounded in</h4>
                  <div className="space-y-2">
                    {reflection.grounded_concepts.map((concept, i) => (
                      <div key={i} className="flex items-start gap-2">
                        <div className="w-1 h-1 rounded-full bg-sage mt-2 flex-shrink-0" />
                        <div>
                          <span className="text-sm font-medium text-forest">
                            {typeof concept === 'string' ? concept : concept.name || concept.concept || ''}
                          </span>
                          {typeof concept === 'object' && concept.source && (
                            <span className="text-xs text-muted ml-2">— {concept.source}</span>
                          )}
                          {typeof concept === 'object' && concept.relevance_score != null && (
                            <span className="text-xs text-muted ml-2">({(concept.relevance_score * 100).toFixed(0)}% relevant)</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Footer */}
              <div className="flex items-center justify-between pt-3 border-t border-border">
                <div className="flex items-center gap-4">
                  <span className="text-xs text-muted">{formatDate(reflection.created_at)}</span>
                  {reflection.confidence_score != null && (
                    <span className="text-xs text-muted">Confidence: {(reflection.confidence_score * 100).toFixed(0)}%</span>
                  )}
                </div>
              </div>

              {/* Disclaimer */}
              <p className="text-[10px] text-muted/60 mt-3">
                This reflection is generated for contemplation and is not professional advice.
              </p>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

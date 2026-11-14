import { useState, useEffect, useMemo } from 'react';
import { conceptsAPI } from '../api/client';
import { Library, Search, ChevronDown, ChevronUp, BookOpen } from 'lucide-react';
import EmptyState from '../components/common/EmptyState';

export default function Concepts() {
  const [concepts, setConcepts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [expandedId, setExpandedId] = useState(null);
  const [expandedData, setExpandedData] = useState({});
  const [loadingDetail, setLoadingDetail] = useState(null);

  useEffect(() => {
    conceptsAPI.list()
      .then(setConcepts)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  // Derive categories dynamically from actual API data
  const categories = useMemo(() => {
    const cats = [...new Set(concepts.map(c => c.category).filter(Boolean))];
    return cats.sort();
  }, [concepts]);

  const filtered = useMemo(() => {
    let list = concepts;
    if (categoryFilter !== 'all') {
      list = list.filter(c => c.category === categoryFilter);
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(c =>
        (c.name || '').toLowerCase().includes(q) ||
        (c.definition || '').toLowerCase().includes(q)
      );
    }
    return list;
  }, [concepts, categoryFilter, search]);

  const handleExpand = async (conceptId) => {
    if (expandedId === conceptId) {
      setExpandedId(null);
      return;
    }
    setExpandedId(conceptId);
    if (!expandedData[conceptId]) {
      setLoadingDetail(conceptId);
      try {
        const detail = await conceptsAPI.get(conceptId);
        setExpandedData(prev => ({ ...prev, [conceptId]: detail }));
      } catch {}
      setLoadingDetail(null);
    }
  };

  if (loading) {
    return <div className="text-muted py-12 text-center font-serif">Loading concepts...</div>;
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="font-serif text-2xl text-charcoal mb-1">Concepts</h1>
        <p className="text-sm text-muted">A library of psychological and philosophical concepts that ground your reflections.</p>
      </div>

      {/* Search & Category Filter */}
      <div className="flex flex-col sm:flex-row gap-3 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search concepts..."
            className="w-full pl-10 pr-4 py-2.5 bg-surface border border-border rounded-lg text-sm text-charcoal placeholder:text-muted/50 focus:border-forest focus:ring-1 focus:ring-forest/20 transition-colors"
          />
        </div>
        <div className="flex gap-1 bg-surface border border-border rounded-lg p-1 flex-wrap">
          <button
            onClick={() => setCategoryFilter('all')}
            className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
              categoryFilter === 'all' ? 'bg-cream text-forest font-medium' : 'text-muted hover:text-charcoal'
            }`}
          >
            All
          </button>
          {categories.map(cat => (
            <button
              key={cat}
              onClick={() => setCategoryFilter(cat)}
              className={`px-3 py-1.5 text-xs rounded-md transition-colors capitalize ${
                categoryFilter === cat ? 'bg-cream text-forest font-medium' : 'text-muted hover:text-charcoal'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Concepts List */}
      {filtered.length === 0 ? (
        <EmptyState
          icon={Library}
          title="No matching concepts"
          description="Try a different search term or category."
        />
      ) : (
        <div className="space-y-3">
          {filtered.map((concept) => {
            const isExpanded = expandedId === concept.id;
            const detail = expandedData[concept.id];
            return (
              <div key={concept.id} className="bg-surface border border-border rounded-lg overflow-hidden">
                <button
                  onClick={() => handleExpand(concept.id)}
                  className="w-full text-left px-5 py-4 hover:bg-ivory/50 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="text-sm font-medium text-charcoal">{concept.name}</h3>
                      <span className="text-[10px] text-sage uppercase tracking-wide">{concept.category}</span>
                      <p className="text-xs text-muted mt-1 line-clamp-2">{concept.definition}</p>
                    </div>
                    <div className="ml-4 flex-shrink-0 mt-1">
                      {isExpanded ? <ChevronUp className="w-4 h-4 text-muted" /> : <ChevronDown className="w-4 h-4 text-muted" />}
                    </div>
                  </div>
                </button>

                {isExpanded && (
                  <div className="px-5 pb-5 border-t border-border pt-4">
                    {loadingDetail === concept.id ? (
                      <p className="text-sm text-muted">Loading details...</p>
                    ) : detail ? (
                      <div className="space-y-4">
                        {detail.explanation && (
                          <div>
                            <h4 className="text-xs font-medium text-muted uppercase tracking-wide mb-1">Explanation</h4>
                            <p className="text-sm text-charcoal leading-relaxed">{detail.explanation}</p>
                          </div>
                        )}
                        {detail.related_patterns && detail.related_patterns.length > 0 && (
                          <div>
                            <h4 className="text-xs font-medium text-muted uppercase tracking-wide mb-1">Related Patterns</h4>
                            <div className="flex flex-wrap gap-2">
                              {detail.related_patterns.map((p, i) => (
                                <span key={i} className="text-xs px-2 py-1 bg-cream text-charcoal rounded">{p}</span>
                              ))}
                            </div>
                          </div>
                        )}
                        {detail.cautions && detail.cautions.length > 0 && (
                          <div>
                            <h4 className="text-xs font-medium text-muted uppercase tracking-wide mb-1">Considerations</h4>
                            <ul className="space-y-1">
                              {detail.cautions.map((c, i) => (
                                <li key={i} className="text-xs text-muted flex gap-2">
                                  <span className="text-sage">•</span> {c}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {detail.source && (
                          <div>
                            <h4 className="text-xs font-medium text-muted uppercase tracking-wide mb-1">Source</h4>
                            <p className="text-xs text-muted italic">{detail.source}</p>
                          </div>
                        )}
                      </div>
                    ) : (
                      <p className="text-sm text-muted">Could not load details.</p>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Dataset notice */}
      <div className="mt-8 pt-4 border-t border-border">
        <p className="text-[10px] text-muted/60">
          This is a 20-concept development/test dataset, not the final production knowledge base.
        </p>
      </div>
    </div>
  );
}

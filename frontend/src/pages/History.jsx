import { useState, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { entriesAPI } from '../api/client';
import { Search, BookOpen, FileText } from 'lucide-react';
import EmptyState from '../components/common/EmptyState';

function formatDate(dateStr) {
  return new Date(dateStr).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
}

export default function History() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all'); // 'all', 'saved', 'drafts'

  useEffect(() => {
    entriesAPI.list('limit=100')
      .then(setEntries)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    let list = entries;
    if (filter === 'saved') list = list.filter(e => !e.is_draft);
    if (filter === 'drafts') list = list.filter(e => e.is_draft);
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(e =>
        (e.title || '').toLowerCase().includes(q) ||
        (e.content || '').toLowerCase().includes(q)
      );
    }
    return list;
  }, [entries, filter, search]);

  if (loading) {
    return <div className="text-muted py-12 text-center font-serif">Loading history...</div>;
  }

  return (
    <div>
      <h1 className="font-serif text-2xl text-charcoal mb-6">History</h1>

      {/* Search & Filter */}
      <div className="flex flex-col sm:flex-row gap-3 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search entries..."
            className="w-full pl-10 pr-4 py-2.5 bg-surface border border-border rounded-lg text-sm text-charcoal placeholder:text-muted/50 focus:border-forest focus:ring-1 focus:ring-forest/20 transition-colors"
          />
        </div>
        <div className="flex gap-1 bg-surface border border-border rounded-lg p-1">
          {[['all', 'All'], ['saved', 'Saved'], ['drafts', 'Drafts']].map(([key, label]) => (
            <button
              key={key}
              onClick={() => setFilter(key)}
              className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                filter === key
                  ? 'bg-cream text-forest font-medium'
                  : 'text-muted hover:text-charcoal'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Entries List */}
      {filtered.length === 0 ? (
        <EmptyState
          icon={BookOpen}
          title={search ? 'No matching entries' : 'No entries yet'}
          description={search ? 'Try a different search term.' : 'Start writing in your journal to see your entries here.'}
          action={
            !search && (
              <Link to="/journal" className="text-sm text-forest hover:underline">Start writing</Link>
            )
          }
        />
      ) : (
        <div className="divide-y divide-border">
          {filtered.map((entry) => (
            <Link
              key={entry.id}
              to={`/journal?id=${entry.id}`}
              className="block py-4 group hover:bg-surface -mx-4 px-4 rounded-lg transition-colors"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="text-sm font-medium text-charcoal group-hover:text-forest transition-colors truncate">
                      {entry.title || 'Untitled'}
                    </h3>
                    {entry.is_draft && (
                      <span className="text-[10px] text-sage bg-cream px-1.5 py-0.5 rounded flex-shrink-0">Draft</span>
                    )}
                    {entry.analysis && (
                      <span className="text-[10px] text-forest bg-forest/10 px-1.5 py-0.5 rounded flex-shrink-0">Analyzed</span>
                    )}
                  </div>
                  <p className="text-xs text-muted line-clamp-2">
                    {entry.content?.substring(0, 150) || ''}
                  </p>
                </div>
                <div className="flex flex-col items-end gap-1 flex-shrink-0">
                  <span className="text-xs text-muted">{formatDate(entry.created_at)}</span>
                  {entry.mood && (
                    <span className="text-[10px] text-muted">{entry.mood}</span>
                  )}
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

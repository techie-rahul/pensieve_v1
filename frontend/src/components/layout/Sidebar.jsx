import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import {
  LayoutDashboard,
  PenLine,
  BookOpen,
  TrendingUp,
  Lightbulb,
  Library,
  LogOut,
  X,
  ShieldCheck,
} from 'lucide-react';

const mainNavItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/journal', label: 'Journal', icon: PenLine },
  { path: '/history', label: 'History', icon: BookOpen },
];

const insightNavItems = [
  { path: '/patterns', label: 'Patterns', icon: TrendingUp },
  { path: '/reflections', label: 'Reflections', icon: Lightbulb },
  { path: '/concepts', label: 'Concepts', icon: Library },
];

export default function Sidebar({ isOpen, onClose }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const userInitials = user?.name
    ? user.name
        .split(' ')
        .map((part) => part[0])
        .join('')
        .slice(0, 2)
        .toUpperCase()
    : 'P';

  const renderNavLink = ({ path, label, icon: Icon }) => (
    <NavLink
      key={path}
      to={path}
      end={path === '/'}
      onClick={onClose}
      className={({ isActive }) =>
        `flex items-center gap-3 px-3 py-2 rounded-lg text-xs tracking-wide transition-all duration-150 ${
          isActive
            ? 'bg-cream/90 text-forest font-medium border-l-[3px] border-forest shadow-xs'
            : 'text-muted hover:text-charcoal hover:bg-ivory/70'
        }`
      }
    >
      <Icon className="w-4 h-4" strokeWidth={1.8} />
      <span>{label}</span>
    </NavLink>
  );

  const sidebarContent = (
    <div className="flex flex-col h-full bg-surface select-none">
      {/* Brand Header */}
      <div className="px-5 pt-6 pb-5 border-b border-border/70 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-cream/80 border border-border flex items-center justify-center text-forest font-serif font-semibold text-base shadow-2xs">
            P
          </div>
          <div>
            <h1 className="font-serif text-xl font-medium text-charcoal tracking-tight leading-none">
              Pensieve
            </h1>
            <p className="text-[10px] text-muted/80 tracking-wider font-sans uppercase mt-1">
              Reflective Journal
            </p>
          </div>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="lg:hidden text-muted hover:text-charcoal p-1.5 rounded-md hover:bg-ivory transition-colors"
            aria-label="Close navigation"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 px-3 py-5 space-y-6 overflow-y-auto">
        {/* Main Writing Section */}
        <div>
          <p className="px-3 text-[10px] font-semibold uppercase tracking-wider text-muted/70 mb-2">
            Writing
          </p>
          <div className="space-y-1">
            {mainNavItems.map(renderNavLink)}
          </div>
        </div>

        {/* Intelligence / Insights Section */}
        <div>
          <div className="border-t border-border/50 pt-4 mb-2">
            <p className="px-3 text-[10px] font-semibold uppercase tracking-wider text-muted/70 mb-2">
              Insights & Signals
            </p>
          </div>
          <div className="space-y-1">
            {insightNavItems.map(renderNavLink)}
          </div>
        </div>
      </nav>

      {/* Footer & User Section */}
      <div className="p-4 border-t border-border/80 bg-surface">
        {/* Private Status Pill */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-ivory border border-border/70 mb-3 text-[11px] text-muted">
          <ShieldCheck className="w-3.5 h-3.5 text-forest flex-shrink-0" />
          <span className="font-medium text-charcoal/80">Private journal</span>
        </div>

        {user && (
          <div className="flex items-center gap-2.5 px-1 mb-3">
            <div className="w-7 h-7 rounded-full bg-cream border border-border text-forest font-serif text-xs font-medium flex items-center justify-center flex-shrink-0">
              {userInitials}
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-medium text-charcoal truncate">{user.name || 'Journaler'}</p>
              <p className="text-[11px] text-muted truncate">{user.email}</p>
            </div>
          </div>
        )}

        <button
          onClick={handleLogout}
          className="flex items-center gap-2 px-2.5 py-1.5 text-xs text-muted hover:text-charcoal transition-colors w-full rounded-md hover:bg-ivory"
        >
          <LogOut className="w-3.5 h-3.5" strokeWidth={1.5} />
          Sign out
        </button>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden lg:flex flex-col w-[230px] min-w-[230px] h-screen bg-surface border-r border-border fixed left-0 top-0 z-30">
        {sidebarContent}
      </aside>

      {/* Mobile overlay drawer */}
      {isOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="fixed inset-0 bg-charcoal/20 backdrop-blur-[1px] transition-opacity"
            onClick={onClose}
          />
          <aside className="fixed left-0 top-0 h-full w-[260px] bg-surface border-r border-border shadow-xl z-50">
            {sidebarContent}
          </aside>
        </div>
      )}
    </>
  );
}

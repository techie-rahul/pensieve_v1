import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import { Menu } from 'lucide-react';

export default function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-ivory">
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* Mobile header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 bg-surface border-b border-border px-4 py-3 z-40 flex items-center">
        <button onClick={() => setSidebarOpen(true)} className="text-muted hover:text-charcoal">
          <Menu className="w-5 h-5" />
        </button>
        <h1 className="font-serif text-lg font-medium text-charcoal ml-4">Pensieve</h1>
      </div>

      {/* Main content */}
      <main className="lg:ml-[230px] min-h-screen pt-14 lg:pt-0">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 py-6 sm:py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

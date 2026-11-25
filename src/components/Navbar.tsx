export type NavTab = 'overview' | 'analyze' | 'benchmarks' | 'history';

interface NavbarProps {
  activeTab: NavTab;
  onNavigate: (tab: NavTab) => void;
  onPrimaryAction: () => void;
}

const NAV_ITEMS: Array<{ id: NavTab; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'analyze', label: 'Analyze' },
  { id: 'benchmarks', label: 'Benchmarks' },
  { id: 'history', label: 'History' },
];

export function Navbar({ activeTab, onNavigate, onPrimaryAction }: NavbarProps) {
  return (
    <header className="sticky top-0 z-40 border-b border-slate-800/70 bg-[#050814]/85 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1380px] items-center justify-between gap-8 px-6 py-3.5">
        {/* Zone 1: Single-line brand wordmark */}
        <button
          type="button"
          onClick={() => onNavigate('overview')}
          className="font-display text-lg font-bold tracking-tight text-white whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded-md px-1"
        >
          SpectraTrace
        </button>

        {/* Zone 2: 4 concise single-line text nav links */}
        <nav className="flex items-center gap-6 overflow-x-auto" aria-label="Primary navigation">
          {NAV_ITEMS.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onNavigate(item.id)}
                className={`relative py-2 text-sm font-medium transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded-sm ${
                  isActive ? 'text-white' : 'text-slate-400 hover:text-slate-100'
                }`}
              >
                {item.label}
                {isActive && (
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-0 -bottom-[1px] h-[2px] rounded-full bg-gradient-to-r from-blue-500 to-violet-500"
                  />
                )}
              </button>
            );
          })}
        </nav>

        {/* Zone 3: 1 primary action */}
        <button
          type="button"
          onClick={onPrimaryAction}
          className="rounded-lg bg-gradient-to-r from-blue-600 to-violet-600 px-4 py-2 text-xs font-semibold text-white shadow-sm transition-opacity hover:opacity-95 active:scale-[0.99] whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
        >
          Analyze Image
        </button>
      </div>
    </header>
  );
}

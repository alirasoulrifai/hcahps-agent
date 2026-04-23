import { useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard, Hospital, Ticket, CalendarClock,
  Package, Search, Menu, X
} from 'lucide-react'

const navItems = [
  { to: '/',               icon: LayoutDashboard, label: 'Dashboard'       },
  { to: '/installed-base', icon: Hospital,        label: 'Installed Base'  },
  { to: '/tickets',        icon: Ticket,          label: 'Service Tickets' },
  { to: '/maintenance',    icon: CalendarClock,   label: 'PM Calendar'     },
  { to: '/spare-parts',    icon: Package,         label: 'Spare Parts'     },
]

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [search, setSearch] = useState('')
  const navigate = useNavigate()

  function handleSearch(e) {
    e.preventDefault()
    if (search.trim()) {
      navigate(`/installed-base?search=${encodeURIComponent(search.trim())}`)
    }
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <aside
        className="flex-shrink-0 flex flex-col transition-all duration-200"
        style={{ width: sidebarOpen ? 240 : 64, background: '#00646E', minHeight: '100vh' }}
      >
        <div className="flex items-center gap-3 px-4 py-5 border-b border-white/10">
          <div className="w-8 h-8 rounded-lg bg-white/20 flex items-center justify-center flex-shrink-0">
            <span className="text-white font-bold text-sm">AS</span>
          </div>
          {sidebarOpen && (
            <div className="overflow-hidden">
              <div className="text-white font-bold text-sm leading-tight whitespace-nowrap">Al Shatta</div>
              <div className="text-white/60 text-xs leading-tight whitespace-nowrap">Medical Equipment CRM</div>
            </div>
          )}
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="ml-auto text-white/70 hover:text-white transition-colors">
            {sidebarOpen ? <X size={16} /> : <Menu size={16} />}
          </button>
        </div>
        <nav className="flex-1 py-4 px-2 space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all text-sm font-medium ${
                  isActive ? 'bg-white/20 text-white' : 'text-white/70 hover:bg-white/10 hover:text-white'
                }`
              }
            >
              <Icon size={18} className="flex-shrink-0" />
              {sidebarOpen && <span className="whitespace-nowrap">{label}</span>}
            </NavLink>
          ))}
        </nav>
        {sidebarOpen && (
          <div className="px-4 py-4 border-t border-white/10">
            <div className="text-white/40 text-xs leading-relaxed">
              Powered by<br />
              <span className="text-white/60 font-medium">Siemens Healthineers</span>
            </div>
          </div>
        )}
      </aside>

      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-4 flex-shrink-0">
          <div className="flex-1">
            <form onSubmit={handleSearch} className="relative max-w-md">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search by hospital name or serial number…"
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="w-full pl-9 pr-4 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none bg-gray-50"
                style={{ '--tw-ring-color': '#00646E' }}
              />
            </form>
          </div>
          <div className="flex items-center gap-2">
            <div className="text-right">
              <div className="text-xs font-semibold text-gray-800">Technical Engineering</div>
              <div className="text-xs text-gray-500">Al Shatta — Siemens Healthineers</div>
            </div>
            <div className="w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-bold" style={{ background: '#00646E' }}>
              TE
            </div>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-6" style={{ background: '#E6E9EB' }}>
          <Outlet context={{ searchQuery: search, setSearchQuery: setSearch }} />
        </main>
      </div>
    </div>
  )
}

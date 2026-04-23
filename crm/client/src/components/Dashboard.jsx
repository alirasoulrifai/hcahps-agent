import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { dashboardApi } from '../utils/api'
import {
  Activity, AlertTriangle, CalendarClock, Package,
  Clock, CheckCircle2, Wrench, ChevronRight, RefreshCw
} from 'lucide-react'

const URGENCY_COLORS = {
  Critical: { bg: '#FEF2F2', text: '#DC2626', border: '#FCA5A5' },
  High:     { bg: '#FFF7ED', text: '#EA580C', border: '#FDBA74' },
  Medium:   { bg: '#FEFCE8', text: '#CA8A04', border: '#FDE047' },
  Low:      { bg: '#F0FDF4', text: '#16A34A', border: '#86EFAC' },
}

const PM_STATUS_COLORS = {
  Overdue:   { bg: '#FEF2F2', text: '#DC2626' },
  Scheduled: { bg: '#E6F4F5', text: '#00646E' },
  Completed: { bg: '#F0FDF4', text: '#16A34A' },
}

function UrgencyBadge({ urgency }) {
  const c = URGENCY_COLORS[urgency] || URGENCY_COLORS.Low
  return (
    <span className="badge" style={{ background: c.bg, color: c.text, border: `1px solid ${c.border}` }}>
      {urgency === 'Critical' && <span className="mr-1">●</span>}
      {urgency}
    </span>
  )
}

function StatCard({ icon: Icon, label, value, sub, color, onClick }) {
  return (
    <button
      onClick={onClick}
      className="card text-left w-full hover:shadow-md transition-shadow"
      style={{ borderLeft: `4px solid ${color}` }}
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm text-gray-500 font-medium">{label}</div>
          <div className="text-3xl font-bold mt-1" style={{ color }}>{value}</div>
          {sub && <div className="text-xs text-gray-400 mt-1">{sub}</div>}
        </div>
        <div className="p-2 rounded-lg" style={{ background: color + '18' }}>
          <Icon size={22} style={{ color }} />
        </div>
      </div>
    </button>
  )
}

export default function Dashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()
  const today = new Date().toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })

  async function load() {
    setLoading(true)
    try {
      const d = await dashboardApi.get()
      setData(d)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <RefreshCw size={28} className="animate-spin text-petrol-500" style={{ color: '#00646E' }} />
    </div>
  )

  return (
    <div className="space-y-6">
      {/* Page title */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Operations Dashboard</h1>
          <p className="text-sm text-gray-500 mt-0.5">{today}</p>
        </div>
        <button onClick={load} className="btn-ghost text-sm">
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>

      {/* Alert banners */}
      {data.pmOverdue > 0 && (
        <div className="rounded-lg px-4 py-3 flex items-center gap-3 text-sm font-medium"
             style={{ background: '#FEF2F2', border: '1px solid #FCA5A5', color: '#DC2626' }}>
          <AlertTriangle size={16} />
          {data.pmOverdue} preventive maintenance schedule{data.pmOverdue > 1 ? 's are' : ' is'} overdue — immediate action required.
          <button onClick={() => navigate('/maintenance')} className="ml-auto underline text-xs">View PM Calendar →</button>
        </div>
      )}
      {data.criticalStock > 0 && (
        <div className="rounded-lg px-4 py-3 flex items-center gap-3 text-sm font-medium"
             style={{ background: '#FFF7ED', border: '1px solid #FDBA74', color: '#EA580C' }}>
          <Package size={16} />
          {data.criticalStock} critical spare part{data.criticalStock > 1 ? 's are' : ' is'} out of stock.
          <button onClick={() => navigate('/spare-parts')} className="ml-auto underline text-xs">View Inventory →</button>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Activity}     label="Active Machines"   value={data.totalMachines} sub="Siemens fleet"             color="#00646E" onClick={() => navigate('/installed-base')} />
        <StatCard icon={Wrench}       label="Open Tickets"      value={data.openTickets}   sub={`${data.urgentTickets} urgent`}          color={data.urgentTickets > 0 ? '#DC2626' : '#00646E'} onClick={() => navigate('/tickets')} />
        <StatCard icon={CalendarClock}label="PM Due (30 days)"  value={data.pmDue + data.pmOverdue} sub={data.pmOverdue > 0 ? `${data.pmOverdue} overdue` : 'On schedule'} color={data.pmOverdue > 0 ? '#EA580C' : '#00646E'} onClick={() => navigate('/maintenance')} />
        <StatCard icon={Package}      label="Low Stock Parts"   value={data.lowStock}      sub={data.criticalStock > 0 ? `${data.criticalStock} out of stock` : 'Parts level'} color={data.criticalStock > 0 ? '#DC2626' : data.lowStock > 0 ? '#EA580C' : '#00646E'} onClick={() => navigate('/spare-parts')} />
      </div>

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Urgent service calls */}
        <div className="lg:col-span-2 card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-bold text-gray-900 flex items-center gap-2">
              <AlertTriangle size={16} style={{ color: '#DC2626' }} />
              Urgent Service Calls
            </h2>
            <button onClick={() => navigate('/tickets')} className="text-xs font-medium hover:underline" style={{ color: '#00646E' }}>
              View all tickets →
            </button>
          </div>
          {data.urgentList.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <CheckCircle2 size={32} className="mx-auto mb-2 text-green-400" />
              <p className="text-sm">No urgent service calls</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table>
                <thead>
                  <tr>
                    <th>Ticket</th>
                    <th>Hospital</th>
                    <th>Machine</th>
                    <th>Urgency</th>
                    <th>Status</th>
                    <th>Assigned</th>
                  </tr>
                </thead>
                <tbody>
                  {data.urgentList.map(t => (
                    <tr key={t.id}>
                      <td>
                        <div className="font-medium text-gray-900 text-xs">#{t.id}</div>
                        <div className="text-gray-500 text-xs max-w-[160px] truncate">{t.title}</div>
                      </td>
                      <td className="text-xs">{t.hospital_name}</td>
                      <td className="text-xs">
                        <span className="font-medium">{t.machine_type}</span>
                        <br /><span className="text-gray-400">{t.machine_model}</span>
                      </td>
                      <td><UrgencyBadge urgency={t.urgency} /></td>
                      <td>
                        <span className="badge" style={{
                          background: t.status === 'In Progress' ? '#EFF6FF' : '#F0FDF4',
                          color: t.status === 'In Progress' ? '#2563EB' : '#16A34A'
                        }}>{t.status}</span>
                      </td>
                      <td className="text-xs text-gray-500">{t.assigned_to || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Right column */}
        <div className="space-y-4">
          {/* Today's PM */}
          <div className="card">
            <h2 className="font-bold text-gray-900 flex items-center gap-2 mb-3">
              <Clock size={16} style={{ color: '#00646E' }} />
              Today's PM Schedule
            </h2>
            {data.todayPM.length === 0 ? (
              <p className="text-sm text-gray-400 py-4 text-center">No PM scheduled for today</p>
            ) : (
              <div className="space-y-2">
                {data.todayPM.map(s => (
                  <div key={s.id} className="p-2 rounded-lg border text-xs" style={{ borderColor: '#E5E7EB', background: '#F9FAFB' }}>
                    <div className="font-semibold text-gray-800">{s.hospital_name}</div>
                    <div className="text-gray-500">{s.machine_type} — {s.machine_model}</div>
                    <div className="mt-1">
                      <span className="badge" style={{ background: '#E6F4F5', color: '#00646E' }}>{s.type}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Upcoming PM */}
          <div className="card">
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-bold text-gray-900 flex items-center gap-2">
                <CalendarClock size={16} style={{ color: '#00646E' }} />
                PM Due Soon
              </h2>
              <button onClick={() => navigate('/maintenance')} className="text-xs font-medium hover:underline" style={{ color: '#00646E' }}>
                Full calendar →
              </button>
            </div>
            {data.upcomingPM.length === 0 ? (
              <p className="text-sm text-gray-400 py-2 text-center">No upcoming PM in 30 days</p>
            ) : (
              <div className="space-y-2">
                {data.upcomingPM.map(s => {
                  const c = PM_STATUS_COLORS[s.status] || PM_STATUS_COLORS.Scheduled
                  return (
                    <div key={s.id} className="flex items-center gap-3 py-1.5 border-b border-gray-100 last:border-0">
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-semibold text-gray-800 truncate">{s.hospital_name}</div>
                        <div className="text-xs text-gray-500 truncate">{s.machine_type} — {s.machine_model}</div>
                      </div>
                      <div className="text-right flex-shrink-0">
                        <span className="badge text-xs" style={{ background: c.bg, color: c.text }}>{s.status}</span>
                        <div className="text-xs text-gray-400 mt-0.5">
                          {new Date(s.scheduled_date).toLocaleDateString('en-GB', { day:'2-digit', month:'short' })}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

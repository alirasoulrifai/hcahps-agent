import { useEffect, useState } from 'react'
import { maintenanceApi, hospitalsApi, machinesApi } from '../utils/api'
import { exportMaintenanceReport } from '../utils/pdf'
import { Plus, Pencil, Trash2, FileDown, RefreshCw, CheckCircle2, AlertTriangle, CalendarDays } from 'lucide-react'

const PM_TYPES = ['Bi-Annual PM','Annual PM','Corrective']
const PM_STATUSES = ['Scheduled','Completed','Overdue']

const STATUS_STYLE = {
  Scheduled: { bg: '#E6F4F5', text: '#00646E', border: '#99D3D7' },
  Completed: { bg: '#F0FDF4', text: '#16A34A', border: '#86EFAC' },
  Overdue:   { bg: '#FEF2F2', text: '#DC2626', border: '#FCA5A5' },
}

const TYPE_ICONS = { 'Bi-Annual PM': '📋', 'Annual PM': '📅', 'Corrective': '🔧' }

const emptyForm = { machine_id: '', hospital_id: '', scheduled_date: '', type: 'Bi-Annual PM', status: 'Scheduled', notes: '' }

function daysUntil(dateStr) {
  const d = Math.ceil((new Date(dateStr) - Date.now()) / 86400000)
  if (d < 0)  return <span style={{ color: '#DC2626' }}>Overdue {Math.abs(d)}d</span>
  if (d === 0) return <span style={{ color: '#EA580C' }}>Today</span>
  if (d <= 7)  return <span style={{ color: '#EA580C' }}>In {d}d</span>
  return <span className="text-gray-500">In {d}d</span>
}

export default function PMCalendar() {
  const [schedules, setSchedules] = useState([])
  const [hospitals, setHospitals] = useState([])
  const [machines, setMachines]   = useState([])
  const [loading, setLoading]     = useState(true)
  const [tab, setTab]             = useState('upcoming')
  const [filterStatus, setFilterStatus] = useState('')
  const [filterType, setFilterType]     = useState('')
  const [modal, setModal]               = useState(null)
  const [delConfirm, setDelConfirm]     = useState(null)
  const [calMonth, setCalMonth]         = useState(new Date())

  async function load() {
    setLoading(true)
    try {
      const [s, h, m] = await Promise.all([maintenanceApi.getAll(), hospitalsApi.getAll(), machinesApi.getAll()])
      setSchedules(s); setHospitals(h); setMachines(m)
    } finally { setLoading(false) }
  }
  useEffect(() => { load() }, [])

  async function saveSchedule(data) {
    if (data.id) await maintenanceApi.update(data.id, data)
    else await maintenanceApi.create(data)
    setModal(null); load()
  }

  async function completeSchedule(s) {
    await maintenanceApi.update(s.id, { ...s, status: 'Completed' })
    load()
  }

  const filtered = schedules.filter(s => {
    const isPast = ['Completed','Overdue'].includes(s.status) && tab === 'past'
    const isUpcoming = ['Scheduled','Overdue'].includes(s.status) && tab === 'upcoming'
    const isCompleted = s.status === 'Completed' && tab === 'past'
    const tabOk = tab === 'upcoming' ? ['Scheduled','Overdue'].includes(s.status) : ['Completed'].includes(s.status)
    return tabOk && (!filterStatus || s.status === filterStatus) && (!filterType || s.type === filterType)
  })

  const overdueCount = schedules.filter(s => s.status === 'Overdue').length
  const upcomingCount = schedules.filter(s => s.status === 'Scheduled').length

  // Calendar helpers
  const year = calMonth.getFullYear()
  const month = calMonth.getMonth()
  const firstDay = new Date(year, month, 1).getDay()
  const daysInMonth = new Date(year, month + 1, 0).getDate()
  const calDates = []
  for (let i = 0; i < firstDay; i++) calDates.push(null)
  for (let d = 1; d <= daysInMonth; d++) calDates.push(d)

  function getSchedulesForDay(day) {
    if (!day) return []
    const dateStr = `${year}-${String(month+1).padStart(2,'0')}-${String(day).padStart(2,'0')}`
    return schedules.filter(s => s.scheduled_date === dateStr)
  }

  if (loading) return <div className="flex items-center justify-center h-64"><RefreshCw size={28} className="animate-spin" style={{ color: '#00646E' }} /></div>

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-gray-900">PM Calendar</h1>
          <p className="text-sm text-gray-500">{overdueCount} overdue · {upcomingCount} scheduled</p>
        </div>
        <div className="flex gap-2 flex-wrap">
          <button onClick={() => exportMaintenanceReport(schedules)} className="btn-ghost"><FileDown size={14} />Export PDF</button>
          <button onClick={() => setModal(emptyForm)} className="btn-primary"><Plus size={14} />Schedule PM</button>
        </div>
      </div>

      {overdueCount > 0 && (
        <div className="rounded-lg px-4 py-3 flex items-center gap-3 text-sm font-medium"
             style={{ background: '#FEF2F2', border: '1px solid #FCA5A5', color: '#DC2626' }}>
          <AlertTriangle size={16} />
          {overdueCount} maintenance schedule{overdueCount > 1 ? 's are' : ' is'} overdue — please reschedule immediately.
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        {/* Calendar */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <button onClick={() => setCalMonth(new Date(year, month - 1, 1))} className="p-1 rounded hover:bg-gray-100 text-gray-500">‹</button>
            <h2 className="font-bold text-gray-900 text-sm">
              {calMonth.toLocaleDateString('en-GB', { month: 'long', year: 'numeric' })}
            </h2>
            <button onClick={() => setCalMonth(new Date(year, month + 1, 1))} className="p-1 rounded hover:bg-gray-100 text-gray-500">›</button>
          </div>
          <div className="grid grid-cols-7 gap-0.5 text-center">
            {['Su','Mo','Tu','We','Th','Fr','Sa'].map(d => (
              <div key={d} className="text-xs font-bold text-gray-400 py-1">{d}</div>
            ))}
            {calDates.map((day, i) => {
              const daySchedules = getSchedulesForDay(day)
              const hasOverdue = daySchedules.some(s => s.status === 'Overdue')
              const hasScheduled = daySchedules.some(s => s.status === 'Scheduled')
              const hasCompleted = daySchedules.some(s => s.status === 'Completed')
              const today = new Date()
              const isToday = day === today.getDate() && month === today.getMonth() && year === today.getFullYear()
              return (
                <div key={i} className={`relative text-xs py-1.5 rounded text-center ${day ? 'cursor-default' : ''} ${isToday ? 'font-bold' : ''}`}
                     style={{
                       background: isToday ? '#00646E' : 'transparent',
                       color: isToday ? 'white' : day ? '#1A1A1A' : 'transparent',
                     }}>
                  {day || ''}
                  {daySchedules.length > 0 && (
                    <div className="flex justify-center gap-0.5 mt-0.5 flex-wrap">
                      {hasOverdue   && <span className="w-1.5 h-1.5 rounded-full bg-red-500 inline-block"></span>}
                      {hasScheduled && <span className="w-1.5 h-1.5 rounded-full inline-block" style={{ background: '#00646E' }}></span>}
                      {hasCompleted && <span className="w-1.5 h-1.5 rounded-full bg-green-500 inline-block"></span>}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
          <div className="mt-4 pt-4 border-t border-gray-100 space-y-1.5">
            {[['Overdue','#DC2626'],['Scheduled','#00646E'],['Completed','#16A34A']].map(([l, c]) => (
              <div key={l} className="flex items-center gap-2 text-xs text-gray-600">
                <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: c }}></span>
                {l}
              </div>
            ))}
          </div>
        </div>

        {/* List */}
        <div className="xl:col-span-2 space-y-4">
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex gap-1 bg-white rounded-xl p-1 shadow-sm">
              {[['upcoming','Upcoming'],['past','Completed']].map(([k,l]) => (
                <button key={k} onClick={() => setTab(k)}
                  className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition-all ${tab === k ? 'text-white' : 'text-gray-500 hover:text-gray-700'}`}
                  style={tab === k ? { background: '#00646E' } : {}}>
                  {l}
                </button>
              ))}
            </div>
            <select className="form-select w-36" value={filterType} onChange={e => setFilterType(e.target.value)}>
              <option value="">All Types</option>
              {PM_TYPES.map(t => <option key={t}>{t}</option>)}
            </select>
          </div>

          <div className="card p-0 overflow-hidden">
            {filtered.length === 0 ? (
              <div className="py-14 text-center text-gray-400">
                <CalendarDays size={36} className="mx-auto mb-3 opacity-30" />
                <p className="text-sm">No maintenance records found</p>
              </div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Hospital</th>
                    <th>Machine</th>
                    <th>Type</th>
                    <th>Scheduled</th>
                    <th>Due In</th>
                    <th>Status</th>
                    <th>Notes</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map(s => {
                    const st = STATUS_STYLE[s.status]
                    return (
                      <tr key={s.id}>
                        <td className="text-sm font-medium">{s.hospital_name}</td>
                        <td className="text-xs">
                          <span className="font-medium">{s.machine_type}</span><br />
                          <span className="text-gray-400">{s.machine_model}</span>
                        </td>
                        <td className="text-xs">
                          <span className="mr-1">{TYPE_ICONS[s.type]}</span>{s.type}
                        </td>
                        <td className="text-xs text-gray-600">
                          {new Date(s.scheduled_date).toLocaleDateString('en-GB', { day:'2-digit', month:'short', year:'numeric' })}
                        </td>
                        <td className="text-xs">{s.status === 'Completed' ? <span className="text-green-600">✓ Done</span> : daysUntil(s.scheduled_date)}</td>
                        <td>
                          <span className="badge" style={{ background: st.bg, color: st.text, border: `1px solid ${st.border}` }}>{s.status}</span>
                        </td>
                        <td className="text-xs text-gray-400 max-w-[120px] truncate">{s.notes || '—'}</td>
                        <td>
                          <div className="flex gap-1">
                            {s.status !== 'Completed' && (
                              <button title="Mark Complete" className="p-1.5 rounded hover:bg-green-50 text-gray-400 hover:text-green-600" onClick={() => completeSchedule(s)}>
                                <CheckCircle2 size={14} />
                              </button>
                            )}
                            <button className="p-1.5 rounded hover:bg-gray-100 text-gray-400 hover:text-blue-600" onClick={() => setModal(s)}>
                              <Pencil size={14} />
                            </button>
                            <button className="p-1.5 rounded hover:bg-red-50 text-gray-400 hover:text-red-600" onClick={() => setDelConfirm(s)}>
                              <Trash2 size={14} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>

      {modal && <PMModal initial={modal} hospitals={hospitals} machines={machines} onSave={saveSchedule} onClose={() => setModal(null)} />}

      {delConfirm && (
        <div className="modal-overlay">
          <div className="modal-box max-w-sm">
            <h3 className="text-lg font-bold mb-2">Delete Schedule</h3>
            <p className="text-sm text-gray-600 mb-6">Delete this PM record for <strong>{delConfirm.hospital_name}</strong>?</p>
            <div className="flex gap-3 justify-end">
              <button className="btn-ghost" onClick={() => setDelConfirm(null)}>Cancel</button>
              <button className="btn-danger" onClick={async () => { await maintenanceApi.remove(delConfirm.id); setDelConfirm(null); load() }}>Delete</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function PMModal({ initial, hospitals, machines, onSave, onClose }) {
  const [form, setForm] = useState({ ...initial })
  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))
  const hospMachines = machines.filter(m => String(m.hospital_id) === String(form.hospital_id))

  return (
    <div className="modal-overlay">
      <div className="modal-box">
        <h3 className="text-lg font-bold text-gray-900 mb-5">{form.id ? 'Edit PM Schedule' : 'Schedule Maintenance'}</h3>
        <div className="form-group">
          <label className="form-label">Hospital *</label>
          <select className="form-select" value={form.hospital_id} onChange={e => { set('hospital_id', e.target.value); set('machine_id', '') }}>
            <option value="">Select hospital</option>
            {hospitals.map(h => <option key={h.id} value={h.id}>{h.name}</option>)}
          </select>
        </div>
        <div className="form-group">
          <label className="form-label">Machine *</label>
          <select className="form-select" value={form.machine_id} onChange={e => set('machine_id', e.target.value)}>
            <option value="">Select machine</option>
            {hospMachines.map(m => <option key={m.id} value={m.id}>{m.type} — {m.model} ({m.serial_number})</option>)}
          </select>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="form-group">
            <label className="form-label">PM Type *</label>
            <select className="form-select" value={form.type} onChange={e => set('type', e.target.value)}>
              {PM_TYPES.map(t => <option key={t}>{t}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Status</label>
            <select className="form-select" value={form.status} onChange={e => set('status', e.target.value)}>
              {PM_STATUSES.map(s => <option key={s}>{s}</option>)}
            </select>
          </div>
        </div>
        <div className="form-group">
          <label className="form-label">Scheduled Date *</label>
          <input type="date" className="form-input" value={form.scheduled_date || ''} onChange={e => set('scheduled_date', e.target.value)} />
        </div>
        <div className="form-group">
          <label className="form-label">Notes</label>
          <textarea className="form-input" rows={2} value={form.notes || ''} onChange={e => set('notes', e.target.value)} placeholder="Scope of work, parts needed…" />
        </div>
        <div className="flex gap-3 justify-end mt-2">
          <button className="btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={() => onSave(form)} disabled={!form.hospital_id || !form.machine_id || !form.scheduled_date}>Save</button>
        </div>
      </div>
    </div>
  )
}

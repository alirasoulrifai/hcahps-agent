import { useEffect, useState } from 'react'
import { ticketsApi, hospitalsApi, machinesApi } from '../utils/api'
import { exportServiceReport, exportTicketsReport } from '../utils/pdf'
import { Plus, Pencil, Trash2, FileDown, RefreshCw, Search, CheckCircle2, AlertTriangle, Clock } from 'lucide-react'

const URGENCIES = ['Critical','High','Medium','Low']
const STATUSES  = ['Open','In Progress','Resolved','Closed']
const ENGINEERS = ['Ahmad Al-Zein','Tariq Mansour','Samer Khalil','Unassigned']

const URGENCY_STYLE = {
  Critical: { bg: '#FEF2F2', text: '#DC2626', border: '#FCA5A5' },
  High:     { bg: '#FFF7ED', text: '#EA580C', border: '#FDBA74' },
  Medium:   { bg: '#FEFCE8', text: '#CA8A04', border: '#FDE047' },
  Low:      { bg: '#F0FDF4', text: '#16A34A', border: '#86EFAC' },
}

const STATUS_STYLE = {
  'Open':        { bg: '#EFF6FF', text: '#2563EB' },
  'In Progress': { bg: '#FFF7ED', text: '#EA580C' },
  'Resolved':    { bg: '#F0FDF4', text: '#16A34A' },
  'Closed':      { bg: '#F3F4F6', text: '#6B7280' },
}

function timeSince(dateStr) {
  if (!dateStr) return '—'
  const diff = Date.now() - new Date(dateStr).getTime()
  const h = Math.floor(diff / 3600000)
  if (h < 24) return `${h}h ago`
  const d = Math.floor(h / 24)
  return `${d}d ago`
}

function timeToResolve(created, resolved) {
  if (!resolved) return null
  const diff = new Date(resolved) - new Date(created)
  const h = Math.floor(diff / 3600000)
  if (h < 24) return `${h}h`
  return `${Math.floor(h / 24)}d ${h % 24}h`
}

const emptyTicket = {
  machine_id: '', hospital_id: '', title: '', description: '',
  urgency: 'Medium', assigned_to: '', status: 'Open', resolution_notes: ''
}

export default function Tickets() {
  const [tickets, setTickets]   = useState([])
  const [hospitals, setHospitals] = useState([])
  const [machines, setMachines]   = useState([])
  const [loading, setLoading]   = useState(true)
  const [activeTab, setActiveTab] = useState('active')
  const [search, setSearch]     = useState('')
  const [filterUrgency, setFilterUrgency] = useState('')
  const [modal, setModal]       = useState(null)
  const [resolveModal, setResolveModal] = useState(null)
  const [delConfirm, setDelConfirm] = useState(null)

  async function load() {
    setLoading(true)
    try {
      const [t, h, m] = await Promise.all([ticketsApi.getAll(), hospitalsApi.getAll(), machinesApi.getAll()])
      setTickets(t); setHospitals(h); setMachines(m)
    } finally { setLoading(false) }
  }
  useEffect(() => { load() }, [])

  const filtered = tickets.filter(t => {
    const q = search.toLowerCase()
    const matchesSearch = !q || t.title.toLowerCase().includes(q) ||
      t.hospital_name.toLowerCase().includes(q) ||
      (t.serial_number || '').toLowerCase().includes(q) ||
      (t.machine_model || '').toLowerCase().includes(q)
    const matchesUrgency = !filterUrgency || t.urgency === filterUrgency
    const isActive = ['Open','In Progress'].includes(t.status)
    const matchesTab = activeTab === 'active' ? isActive : !isActive
    return matchesSearch && matchesUrgency && matchesTab
  })

  const counts = {
    active: tickets.filter(t => ['Open','In Progress'].includes(t.status)).length,
    resolved: tickets.filter(t => ['Resolved','Closed'].includes(t.status)).length,
  }

  async function saveTicket(data) {
    if (data.id) await ticketsApi.update(data.id, data)
    else await ticketsApi.create(data)
    setModal(null)
    load()
  }

  async function resolveTicket(id, notes) {
    await ticketsApi.update(id, { ...tickets.find(t => t.id === id), status: 'Resolved', resolution_notes: notes })
    setResolveModal(null)
    load()
  }

  async function deleteTicket(id) {
    await ticketsApi.remove(id)
    setDelConfirm(null)
    load()
  }

  if (loading) return <div className="flex items-center justify-center h-64"><RefreshCw size={28} className="animate-spin" style={{ color: '#00646E' }} /></div>

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Service Tickets</h1>
          <p className="text-sm text-gray-500">{counts.active} active · {counts.resolved} resolved</p>
        </div>
        <div className="flex gap-2 flex-wrap">
          <button onClick={() => exportTicketsReport(tickets)} className="btn-ghost"><FileDown size={14} />Export PDF</button>
          <button onClick={() => setModal(emptyTicket)} className="btn-orange"><Plus size={14} />New Ticket</button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-white rounded-xl p-1 w-fit shadow-sm">
        {[['active','Active'], ['resolved','Resolved / Closed']].map(([k, l]) => (
          <button
            key={k}
            onClick={() => setActiveTab(k)}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${activeTab === k ? 'text-white' : 'text-gray-500 hover:text-gray-700'}`}
            style={activeTab === k ? { background: '#00646E' } : {}}
          >
            {l} <span className="ml-1 opacity-75">({counts[k]})</span>
          </button>
        ))}
      </div>

      {/* Filters */}
      <div className="flex gap-3 flex-wrap">
        <div className="relative flex-1 min-w-[200px]">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input className="form-input pl-8" placeholder="Search tickets…" value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <select className="form-select w-36" value={filterUrgency} onChange={e => setFilterUrgency(e.target.value)}>
          <option value="">All Urgencies</option>
          {URGENCIES.map(u => <option key={u}>{u}</option>)}
        </select>
      </div>

      {/* Table */}
      <div className="card p-0 overflow-hidden">
        {filtered.length === 0 ? (
          <div className="py-14 text-center text-gray-400">
            <CheckCircle2 size={36} className="mx-auto mb-3 opacity-30" />
            <p className="text-sm">No tickets found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Title</th>
                  <th>Hospital</th>
                  <th>Machine</th>
                  <th>Urgency</th>
                  <th>Status</th>
                  <th>Assigned To</th>
                  <th>Age / TTR</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(t => {
                  const us = URGENCY_STYLE[t.urgency] || URGENCY_STYLE.Low
                  const ss = STATUS_STYLE[t.status] || STATUS_STYLE.Closed
                  const ttr = timeToResolve(t.created_at, t.resolved_at)
                  return (
                    <tr key={t.id}>
                      <td className="text-xs font-mono text-gray-500">#{t.id}</td>
                      <td>
                        <div className="font-medium text-sm text-gray-900 max-w-[200px]">{t.title}</div>
                        {t.description && <div className="text-xs text-gray-400 truncate max-w-[200px]">{t.description}</div>}
                      </td>
                      <td className="text-sm">{t.hospital_name}</td>
                      <td className="text-xs">
                        <span className="font-medium">{t.machine_type}</span><br />
                        <span className="text-gray-400">{t.machine_model}</span><br />
                        <code className="bg-gray-100 px-1 rounded text-xs">{t.serial_number}</code>
                      </td>
                      <td>
                        <span className="badge" style={{ background: us.bg, color: us.text, border: `1px solid ${us.border}` }}>
                          {t.urgency === 'Critical' && '● '}{t.urgency}
                        </span>
                      </td>
                      <td>
                        <span className="badge" style={{ background: ss.bg, color: ss.text }}>{t.status}</span>
                      </td>
                      <td className="text-sm">{t.assigned_to || '—'}</td>
                      <td className="text-xs text-gray-500">
                        {ttr ? <><span className="text-green-600 font-medium">✓ {ttr}</span></> : timeSince(t.created_at)}
                      </td>
                      <td>
                        <div className="flex gap-1">
                          {['Open','In Progress'].includes(t.status) && (
                            <button className="p-1.5 rounded hover:bg-green-50 text-gray-400 hover:text-green-600" title="Resolve" onClick={() => setResolveModal(t)}>
                              <CheckCircle2 size={14} />
                            </button>
                          )}
                          <button className="p-1.5 rounded hover:bg-gray-100 text-gray-400 hover:text-blue-600" onClick={() => setModal(t)}>
                            <Pencil size={14} />
                          </button>
                          <button className="p-1.5 rounded hover:bg-red-50 text-gray-400 hover:text-red-600" onClick={() => setDelConfirm(t)}>
                            <Trash2 size={14} />
                          </button>
                          <button className="p-1.5 rounded hover:bg-gray-100 text-gray-400 hover:text-petrol" title="Export PDF" onClick={() => exportServiceReport(t, t, t)}>
                            <FileDown size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {modal && <TicketModal initial={modal} hospitals={hospitals} machines={machines} onSave={saveTicket} onClose={() => setModal(null)} />}

      {resolveModal && (
        <ResolveModal ticket={resolveModal} onResolve={resolveTicket} onClose={() => setResolveModal(null)} />
      )}

      {delConfirm && (
        <div className="modal-overlay">
          <div className="modal-box max-w-sm">
            <h3 className="text-lg font-bold text-gray-900 mb-2">Delete Ticket</h3>
            <p className="text-sm text-gray-600 mb-6">Delete ticket <strong>#{delConfirm.id}</strong>: "{delConfirm.title}"? This cannot be undone.</p>
            <div className="flex gap-3 justify-end">
              <button className="btn-ghost" onClick={() => setDelConfirm(null)}>Cancel</button>
              <button className="btn-danger" onClick={() => deleteTicket(delConfirm.id)}>Delete</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function TicketModal({ initial, hospitals, machines, onSave, onClose }) {
  const [form, setForm] = useState({ ...initial })
  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))
  const hospMachines = machines.filter(m => String(m.hospital_id) === String(form.hospital_id))

  useEffect(() => {
    if (form.hospital_id && hospMachines.length > 0 && !form.machine_id) {
      set('machine_id', hospMachines[0].id)
    }
  }, [form.hospital_id])

  return (
    <div className="modal-overlay">
      <div className="modal-box">
        <h3 className="text-lg font-bold text-gray-900 mb-5">{form.id ? 'Edit Ticket' : 'New Service Ticket'}</h3>
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
        <div className="form-group">
          <label className="form-label">Title *</label>
          <input className="form-input" value={form.title} onChange={e => set('title', e.target.value)} placeholder="Brief description of issue" />
        </div>
        <div className="form-group">
          <label className="form-label">Description</label>
          <textarea className="form-input" rows={3} value={form.description || ''} onChange={e => set('description', e.target.value)} placeholder="Detailed description…" />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="form-group">
            <label className="form-label">Urgency *</label>
            <select className="form-select" value={form.urgency} onChange={e => set('urgency', e.target.value)}>
              {URGENCIES.map(u => <option key={u}>{u}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Status</label>
            <select className="form-select" value={form.status || 'Open'} onChange={e => set('status', e.target.value)}>
              {STATUSES.map(s => <option key={s}>{s}</option>)}
            </select>
          </div>
        </div>
        <div className="form-group">
          <label className="form-label">Assigned To</label>
          <select className="form-select" value={form.assigned_to || ''} onChange={e => set('assigned_to', e.target.value)}>
            {ENGINEERS.map(e => <option key={e} value={e === 'Unassigned' ? '' : e}>{e}</option>)}
          </select>
        </div>
        <div className="flex gap-3 justify-end mt-2">
          <button className="btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={() => onSave(form)} disabled={!form.hospital_id || !form.machine_id || !form.title}>Save Ticket</button>
        </div>
      </div>
    </div>
  )
}

function ResolveModal({ ticket, onResolve, onClose }) {
  const [notes, setNotes] = useState('')
  return (
    <div className="modal-overlay">
      <div className="modal-box max-w-md">
        <h3 className="text-lg font-bold text-gray-900 mb-1">Resolve Ticket #{ticket.id}</h3>
        <p className="text-sm text-gray-500 mb-4">{ticket.title}</p>
        <div className="form-group">
          <label className="form-label">Resolution Notes</label>
          <textarea className="form-input" rows={4} value={notes} onChange={e => setNotes(e.target.value)} placeholder="Describe what was done to resolve this issue…" />
        </div>
        <div className="flex gap-3 justify-end">
          <button className="btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn-primary" style={{ background: '#16A34A' }} onClick={() => onResolve(ticket.id, notes)}>Mark Resolved</button>
        </div>
      </div>
    </div>
  )
}

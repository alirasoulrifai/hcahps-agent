import { useEffect, useState } from 'react'
import { useOutletContext, useSearchParams } from 'react-router-dom'
import { hospitalsApi, machinesApi } from '../utils/api'
import { exportInstalledBaseReport } from '../utils/pdf'
import {
  Plus, Pencil, Trash2, ChevronDown, ChevronRight,
  Building2, Cpu, FileDown, RefreshCw, Search
} from 'lucide-react'

const MACHINE_TYPES = ['MRI','CT','X-Ray','Ultrasound','Angiography']
const MACHINE_STATUSES = ['active','inactive','under_maintenance']
const STATUS_LABELS = { active: 'Active', inactive: 'Inactive', under_maintenance: 'Under Maintenance' }
const STATUS_COLORS = {
  active:            { bg: '#F0FDF4', text: '#16A34A' },
  inactive:          { bg: '#F3F4F6', text: '#6B7280' },
  under_maintenance: { bg: '#FFF7ED', text: '#EA580C' },
}

function warrantyStatus(dateStr) {
  if (!dateStr) return { label: 'N/A', bg: '#F3F4F6', text: '#6B7280' }
  const exp = new Date(dateStr)
  const now = new Date()
  const diff = (exp - now) / 86400000
  if (diff < 0)   return { label: 'Expired', bg: '#FEF2F2', text: '#DC2626' }
  if (diff < 90)  return { label: 'Expiring', bg: '#FFF7ED', text: '#EA580C' }
  return { label: 'Active', bg: '#F0FDF4', text: '#16A34A' }
}

function MachineTypeIcon({ type }) {
  const icons = { MRI: '🧲', CT: '🔵', 'X-Ray': '☢️', Ultrasound: '🔊', Angiography: '💉' }
  return <span className="text-base">{icons[type] || '⚙️'}</span>
}

const emptyHosp = { name: '', city: '', address: '', phone: '', contact_name: '' }
const emptyMachine = { hospital_id: '', type: 'MRI', model: '', serial_number: '', installation_date: '', warranty_expiry: '', status: 'active', notes: '' }

export default function InstalledBase() {
  const [hospitals, setHospitals] = useState([])
  const [machines, setMachines] = useState([])
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState({})
  const [localSearch, setLocalSearch] = useState('')
  const [filterType, setFilterType] = useState('')
  const [hospModal, setHospModal] = useState(null)
  const [machineModal, setMachineModal] = useState(null)
  const [delConfirm, setDelConfirm] = useState(null)
  const [searchParams] = useSearchParams()

  useEffect(() => {
    const q = searchParams.get('search')
    if (q) setLocalSearch(q)
  }, [searchParams])

  async function load() {
    setLoading(true)
    try {
      const [h, m] = await Promise.all([hospitalsApi.getAll(), machinesApi.getAll()])
      setHospitals(h)
      setMachines(m)
    } finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  const query = localSearch.toLowerCase()
  const filteredHospitals = hospitals.filter(h => {
    const hMatches = h.name.toLowerCase().includes(query) || h.city.toLowerCase().includes(query)
    const hMachines = machines.filter(m => m.hospital_id === h.id)
    const machineMatches = hMachines.some(m =>
      m.serial_number.toLowerCase().includes(query) ||
      m.model.toLowerCase().includes(query) ||
      m.type.toLowerCase().includes(query)
    )
    const typeOk = !filterType || hMachines.some(m => m.type === filterType)
    return (query ? (hMatches || machineMatches) : true) && typeOk
  })

  function toggleExpand(id) {
    setExpanded(p => ({ ...p, [id]: !p[id] }))
  }

  async function saveHospital(data) {
    if (data.id) await hospitalsApi.update(data.id, data)
    else await hospitalsApi.create(data)
    setHospModal(null)
    load()
  }

  async function saveMachine(data) {
    if (data.id) await machinesApi.update(data.id, data)
    else await machinesApi.create(data)
    setMachineModal(null)
    load()
  }

  async function confirmDelete() {
    if (!delConfirm) return
    if (delConfirm.type === 'hospital') await hospitalsApi.remove(delConfirm.id)
    else await machinesApi.remove(delConfirm.id)
    setDelConfirm(null)
    load()
  }

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <RefreshCw size={28} className="animate-spin" style={{ color: '#00646E' }} />
    </div>
  )

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Installed Base</h1>
          <p className="text-sm text-gray-500">{hospitals.length} facilities · {machines.length} machines</p>
        </div>
        <div className="flex gap-2 flex-wrap">
          <button onClick={() => exportInstalledBaseReport(hospitals, machines)} className="btn-ghost">
            <FileDown size={14} /> Export PDF
          </button>
          <button onClick={() => setHospModal(emptyHosp)} className="btn-primary">
            <Plus size={14} /> Add Hospital
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-3 flex-wrap">
        <div className="relative flex-1 min-w-[200px]">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            className="form-input pl-8"
            placeholder="Search hospital, city, serial number…"
            value={localSearch}
            onChange={e => setLocalSearch(e.target.value)}
          />
        </div>
        <select className="form-select w-44" value={filterType} onChange={e => setFilterType(e.target.value)}>
          <option value="">All Machine Types</option>
          {MACHINE_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
      </div>

      {/* Hospital list */}
      <div className="space-y-3">
        {filteredHospitals.length === 0 && (
          <div className="card text-center py-12 text-gray-400">
            <Building2 size={40} className="mx-auto mb-3 opacity-30" />
            <p>No hospitals found</p>
          </div>
        )}
        {filteredHospitals.map(h => {
          const hMachines = machines.filter(m => m.hospital_id === h.id &&
            (!filterType || m.type === filterType) &&
            (!query || m.serial_number.toLowerCase().includes(query) || m.model.toLowerCase().includes(query) ||
             m.type.toLowerCase().includes(query) || h.name.toLowerCase().includes(query) || h.city.toLowerCase().includes(query))
          )
          const isOpen = expanded[h.id]
          return (
            <div key={h.id} className="card p-0 overflow-hidden">
              {/* Hospital header */}
              <div
                className="flex items-center gap-3 px-5 py-4 cursor-pointer hover:bg-gray-50 transition-colors"
                onClick={() => toggleExpand(h.id)}
              >
                <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
                     style={{ background: '#E6F4F5' }}>
                  <Building2 size={18} style={{ color: '#00646E' }} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-semibold text-gray-900">{h.name}</div>
                  <div className="text-sm text-gray-500">{h.city} · {h.phone || 'No phone'} · {h.contact_name || 'No contact'}</div>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-xs font-semibold px-3 py-1 rounded-full" style={{ background: '#E6F4F5', color: '#00646E' }}>
                    {hMachines.length} machine{hMachines.length !== 1 ? 's' : ''}
                  </span>
                  <div className="flex gap-1" onClick={e => e.stopPropagation()}>
                    <button className="p-1.5 rounded hover:bg-gray-100 text-gray-400 hover:text-petrol" onClick={() => setMachineModal({ ...emptyMachine, hospital_id: h.id })}>
                      <Plus size={15} />
                    </button>
                    <button className="p-1.5 rounded hover:bg-gray-100 text-gray-400 hover:text-blue-600" onClick={() => setHospModal(h)}>
                      <Pencil size={15} />
                    </button>
                    <button className="p-1.5 rounded hover:bg-red-50 text-gray-400 hover:text-red-600" onClick={() => setDelConfirm({ type: 'hospital', id: h.id, name: h.name })}>
                      <Trash2 size={15} />
                    </button>
                  </div>
                  {isOpen ? <ChevronDown size={16} className="text-gray-400" /> : <ChevronRight size={16} className="text-gray-400" />}
                </div>
              </div>

              {/* Machines table */}
              {isOpen && (
                <div className="border-t border-gray-100">
                  {hMachines.length === 0 ? (
                    <div className="px-5 py-6 text-center text-sm text-gray-400">
                      No machines — <button className="underline" style={{ color: '#00646E' }} onClick={() => setMachineModal({ ...emptyMachine, hospital_id: h.id })}>add one</button>
                    </div>
                  ) : (
                    <table>
                      <thead>
                        <tr>
                          <th>Type</th>
                          <th>Model</th>
                          <th>Serial Number</th>
                          <th>Installed</th>
                          <th>Warranty</th>
                          <th>Status</th>
                          <th>Notes</th>
                          <th></th>
                        </tr>
                      </thead>
                      <tbody>
                        {hMachines.map(m => {
                          const ws = warrantyStatus(m.warranty_expiry)
                          const sc = STATUS_COLORS[m.status]
                          return (
                            <tr key={m.id}>
                              <td>
                                <div className="flex items-center gap-2">
                                  <MachineTypeIcon type={m.type} />
                                  <span className="font-semibold text-xs">{m.type}</span>
                                </div>
                              </td>
                              <td className="font-medium text-sm">{m.model}</td>
                              <td><code className="text-xs bg-gray-100 px-2 py-0.5 rounded">{m.serial_number}</code></td>
                              <td className="text-xs text-gray-600">
                                {m.installation_date ? new Date(m.installation_date).toLocaleDateString('en-GB', { day:'2-digit', month:'short', year:'numeric' }) : '—'}
                              </td>
                              <td>
                                <span className="badge" style={{ background: ws.bg, color: ws.text }}>
                                  {ws.label}
                                  {m.warranty_expiry && <span className="ml-1 opacity-70 text-xs">
                                    {new Date(m.warranty_expiry).toLocaleDateString('en-GB', { day:'2-digit', month:'short', year:'2-digit' })}
                                  </span>}
                                </span>
                              </td>
                              <td>
                                <span className="badge" style={{ background: sc.bg, color: sc.text }}>{STATUS_LABELS[m.status]}</span>
                              </td>
                              <td className="text-xs text-gray-400 max-w-[140px] truncate">{m.notes || '—'}</td>
                              <td>
                                <div className="flex gap-1">
                                  <button className="p-1 rounded hover:bg-gray-100 text-gray-400 hover:text-blue-600" onClick={() => setMachineModal(m)}>
                                    <Pencil size={13} />
                                  </button>
                                  <button className="p-1 rounded hover:bg-red-50 text-gray-400 hover:text-red-600" onClick={() => setDelConfirm({ type: 'machine', id: m.id, name: `${m.type} — ${m.model}` })}>
                                    <Trash2 size={13} />
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
              )}
            </div>
          )
        })}
      </div>

      {/* Hospital Modal */}
      {hospModal && (
        <HospitalModal
          initial={hospModal}
          onSave={saveHospital}
          onClose={() => setHospModal(null)}
        />
      )}

      {/* Machine Modal */}
      {machineModal && (
        <MachineModal
          initial={machineModal}
          hospitals={hospitals}
          onSave={saveMachine}
          onClose={() => setMachineModal(null)}
        />
      )}

      {/* Delete confirm */}
      {delConfirm && (
        <div className="modal-overlay">
          <div className="modal-box max-w-sm">
            <h3 className="text-lg font-bold text-gray-900 mb-2">Confirm Delete</h3>
            <p className="text-sm text-gray-600 mb-6">
              Delete <strong>{delConfirm.name}</strong>? This action cannot be undone.
              {delConfirm.type === 'hospital' && ' All associated machines, tickets, and maintenance records will also be deleted.'}
            </p>
            <div className="flex gap-3 justify-end">
              <button className="btn-ghost" onClick={() => setDelConfirm(null)}>Cancel</button>
              <button className="btn-danger" onClick={confirmDelete}>Delete</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function HospitalModal({ initial, onSave, onClose }) {
  const [form, setForm] = useState({ ...initial })
  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))
  return (
    <div className="modal-overlay">
      <div className="modal-box">
        <h3 className="text-lg font-bold text-gray-900 mb-5">{form.id ? 'Edit Hospital' : 'Add Hospital'}</h3>
        <div className="form-group">
          <label className="form-label">Hospital / Clinic Name *</label>
          <input className="form-input" value={form.name} onChange={e => set('name', e.target.value)} placeholder="e.g. Damascus University Hospital" />
        </div>
        <div className="form-group">
          <label className="form-label">City *</label>
          <input className="form-input" value={form.city} onChange={e => set('city', e.target.value)} placeholder="e.g. Damascus" />
        </div>
        <div className="form-group">
          <label className="form-label">Address</label>
          <input className="form-input" value={form.address || ''} onChange={e => set('address', e.target.value)} placeholder="Street address" />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="form-group">
            <label className="form-label">Phone</label>
            <input className="form-input" value={form.phone || ''} onChange={e => set('phone', e.target.value)} placeholder="+963-11-…" />
          </div>
          <div className="form-group">
            <label className="form-label">Technical Contact</label>
            <input className="form-input" value={form.contact_name || ''} onChange={e => set('contact_name', e.target.value)} placeholder="Dr. Name" />
          </div>
        </div>
        <div className="flex gap-3 justify-end mt-2">
          <button className="btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={() => onSave(form)} disabled={!form.name || !form.city}>Save Hospital</button>
        </div>
      </div>
    </div>
  )
}

function MachineModal({ initial, hospitals, onSave, onClose }) {
  const [form, setForm] = useState({ ...initial })
  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))
  return (
    <div className="modal-overlay">
      <div className="modal-box">
        <h3 className="text-lg font-bold text-gray-900 mb-5">{form.id ? 'Edit Machine' : 'Add Machine'}</h3>
        <div className="form-group">
          <label className="form-label">Hospital *</label>
          <select className="form-select" value={form.hospital_id} onChange={e => set('hospital_id', e.target.value)}>
            <option value="">Select hospital</option>
            {hospitals.map(h => <option key={h.id} value={h.id}>{h.name}</option>)}
          </select>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="form-group">
            <label className="form-label">Machine Type *</label>
            <select className="form-select" value={form.type} onChange={e => set('type', e.target.value)}>
              {MACHINE_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Status</label>
            <select className="form-select" value={form.status} onChange={e => set('status', e.target.value)}>
              {MACHINE_STATUSES.map(s => <option key={s} value={s}>{STATUS_LABELS[s]}</option>)}
            </select>
          </div>
        </div>
        <div className="form-group">
          <label className="form-label">Model *</label>
          <input className="form-input" value={form.model} onChange={e => set('model', e.target.value)} placeholder="e.g. MAGNETOM Vida 3T" />
        </div>
        <div className="form-group">
          <label className="form-label">Serial Number *</label>
          <input className="form-input" value={form.serial_number} onChange={e => set('serial_number', e.target.value)} placeholder="e.g. MRI-SH-2023-001" />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="form-group">
            <label className="form-label">Installation Date</label>
            <input type="date" className="form-input" value={form.installation_date || ''} onChange={e => set('installation_date', e.target.value)} />
          </div>
          <div className="form-group">
            <label className="form-label">Warranty Expiry</label>
            <input type="date" className="form-input" value={form.warranty_expiry || ''} onChange={e => set('warranty_expiry', e.target.value)} />
          </div>
        </div>
        <div className="form-group">
          <label className="form-label">Notes</label>
          <textarea className="form-input" rows={2} value={form.notes || ''} onChange={e => set('notes', e.target.value)} placeholder="Optional notes" />
        </div>
        <div className="flex gap-3 justify-end mt-2">
          <button className="btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={() => onSave(form)}
            disabled={!form.hospital_id || !form.model || !form.serial_number}>Save Machine</button>
        </div>
      </div>
    </div>
  )
}

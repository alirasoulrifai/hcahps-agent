import { useEffect, useState } from 'react'
import { partsApi } from '../utils/api'
import { exportInventoryReport } from '../utils/pdf'
import { Plus, Pencil, Trash2, FileDown, RefreshCw, Search, Package, AlertTriangle } from 'lucide-react'

const emptyPart = { part_number: '', name: '', description: '', compatible_machines: '', quantity: 0, min_quantity: 1, unit_price: '', supplier: '' }

function stockStatus(qty, minQty) {
  if (qty === 0)       return { label: 'Out of Stock', bg: '#FEF2F2', text: '#DC2626', border: '#FCA5A5' }
  if (qty <= minQty)   return { label: 'Low Stock',    bg: '#FFF7ED', text: '#EA580C', border: '#FDBA74' }
  return                      { label: 'In Stock',     bg: '#F0FDF4', text: '#16A34A', border: '#86EFAC' }
}

export default function SpareParts() {
  const [parts, setParts]     = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch]   = useState('')
  const [filterStatus, setFilterStatus] = useState('')
  const [modal, setModal]     = useState(null)
  const [delConfirm, setDelConfirm] = useState(null)

  async function load() {
    setLoading(true)
    try { setParts(await partsApi.getAll()) }
    finally { setLoading(false) }
  }
  useEffect(() => { load() }, [])

  const filtered = parts.filter(p => {
    const q = search.toLowerCase()
    const matchSearch = !q || p.part_number.toLowerCase().includes(q) ||
      p.name.toLowerCase().includes(q) ||
      (p.compatible_machines || '').toLowerCase().includes(q)
    const st = stockStatus(p.quantity, p.min_quantity).label
    const matchStatus = !filterStatus || st === filterStatus
    return matchSearch && matchStatus
  })

  const outOfStock  = parts.filter(p => p.quantity === 0).length
  const lowStock    = parts.filter(p => p.quantity > 0 && p.quantity <= p.min_quantity).length
  const totalValue  = parts.reduce((acc, p) => acc + (p.quantity * (p.unit_price || 0)), 0)

  async function savePart(data) {
    if (data.id) await partsApi.update(data.id, data)
    else await partsApi.create(data)
    setModal(null); load()
  }

  if (loading) return <div className="flex items-center justify-center h-64"><RefreshCw size={28} className="animate-spin" style={{ color: '#00646E' }} /></div>

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Spare Parts Inventory</h1>
          <p className="text-sm text-gray-500">{parts.length} parts · Est. value: ${totalValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}</p>
        </div>
        <div className="flex gap-2 flex-wrap">
          <button onClick={() => exportInventoryReport(parts)} className="btn-ghost"><FileDown size={14} />Export PDF</button>
          <button onClick={() => setModal(emptyPart)} className="btn-primary"><Plus size={14} />Add Part</button>
        </div>
      </div>

      {/* Alert banners */}
      {outOfStock > 0 && (
        <div className="rounded-lg px-4 py-3 flex items-center gap-3 text-sm font-medium"
             style={{ background: '#FEF2F2', border: '1px solid #FCA5A5', color: '#DC2626' }}>
          <AlertTriangle size={16} />
          <strong>{outOfStock} part{outOfStock > 1 ? 's' : ''}</strong> out of stock — critical restocking required.
        </div>
      )}
      {lowStock > 0 && outOfStock === 0 && (
        <div className="rounded-lg px-4 py-3 flex items-center gap-3 text-sm font-medium"
             style={{ background: '#FFF7ED', border: '1px solid #FDBA74', color: '#EA580C' }}>
          <Package size={16} />
          {lowStock} part{lowStock > 1 ? 's' : ''} running low — review stock levels.
        </div>
      )}

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Out of Stock', value: outOfStock, color: '#DC2626', bg: '#FEF2F2' },
          { label: 'Low Stock',    value: lowStock,   color: '#EA580C', bg: '#FFF7ED' },
          { label: 'Total Parts',  value: parts.length, color: '#00646E', bg: '#E6F4F5' },
        ].map(s => (
          <div key={s.label} className="card text-center py-4" style={{ borderTop: `3px solid ${s.color}` }}>
            <div className="text-2xl font-bold" style={{ color: s.color }}>{s.value}</div>
            <div className="text-xs text-gray-500 mt-0.5">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="flex gap-3 flex-wrap">
        <div className="relative flex-1 min-w-[200px]">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input className="form-input pl-8" placeholder="Search part number, name, compatible machines…" value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <select className="form-select w-40" value={filterStatus} onChange={e => setFilterStatus(e.target.value)}>
          <option value="">All Status</option>
          <option>Out of Stock</option>
          <option>Low Stock</option>
          <option>In Stock</option>
        </select>
      </div>

      {/* Table */}
      <div className="card p-0 overflow-hidden">
        {filtered.length === 0 ? (
          <div className="py-14 text-center text-gray-400">
            <Package size={36} className="mx-auto mb-3 opacity-30" />
            <p className="text-sm">No parts found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <th>Part Number</th>
                  <th>Name</th>
                  <th>Compatible Machines</th>
                  <th>Qty</th>
                  <th>Min Qty</th>
                  <th>Status</th>
                  <th>Unit Price</th>
                  <th>Supplier</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(p => {
                  const st = stockStatus(p.quantity, p.min_quantity)
                  return (
                    <tr key={p.id}>
                      <td><code className="text-xs bg-gray-100 px-2 py-0.5 rounded font-mono">{p.part_number}</code></td>
                      <td>
                        <div className="font-medium text-sm text-gray-900">{p.name}</div>
                        {p.description && <div className="text-xs text-gray-400 max-w-[200px] truncate">{p.description}</div>}
                      </td>
                      <td>
                        <div className="flex flex-wrap gap-1 max-w-[220px]">
                          {(p.compatible_machines || '').split(',').map(m => m.trim()).filter(Boolean).map(m => (
                            <span key={m} className="badge" style={{ background: '#E6F4F5', color: '#00646E', fontSize: '0.7rem' }}>{m}</span>
                          ))}
                        </div>
                      </td>
                      <td className="text-center font-bold text-sm" style={{ color: p.quantity === 0 ? '#DC2626' : '#1A1A1A' }}>
                        {p.quantity}
                      </td>
                      <td className="text-center text-sm text-gray-500">{p.min_quantity}</td>
                      <td>
                        <span className="badge" style={{ background: st.bg, color: st.text, border: `1px solid ${st.border}` }}>{st.label}</span>
                      </td>
                      <td className="text-sm text-gray-700">
                        {p.unit_price ? `$${Number(p.unit_price).toLocaleString()}` : '—'}
                      </td>
                      <td className="text-xs text-gray-500">{p.supplier || '—'}</td>
                      <td>
                        <div className="flex gap-1">
                          <button className="p-1.5 rounded hover:bg-gray-100 text-gray-400 hover:text-blue-600" onClick={() => setModal(p)}>
                            <Pencil size={14} />
                          </button>
                          <button className="p-1.5 rounded hover:bg-red-50 text-gray-400 hover:text-red-600" onClick={() => setDelConfirm(p)}>
                            <Trash2 size={14} />
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

      {modal && <PartModal initial={modal} onSave={savePart} onClose={() => setModal(null)} />}

      {delConfirm && (
        <div className="modal-overlay">
          <div className="modal-box max-w-sm">
            <h3 className="text-lg font-bold mb-2">Delete Part</h3>
            <p className="text-sm text-gray-600 mb-6">Delete <strong>{delConfirm.name}</strong> ({delConfirm.part_number})?</p>
            <div className="flex gap-3 justify-end">
              <button className="btn-ghost" onClick={() => setDelConfirm(null)}>Cancel</button>
              <button className="btn-danger" onClick={async () => { await partsApi.remove(delConfirm.id); setDelConfirm(null); load() }}>Delete</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function PartModal({ initial, onSave, onClose }) {
  const [form, setForm] = useState({ ...initial })
  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))
  return (
    <div className="modal-overlay">
      <div className="modal-box">
        <h3 className="text-lg font-bold text-gray-900 mb-5">{form.id ? 'Edit Spare Part' : 'Add Spare Part'}</h3>
        <div className="grid grid-cols-2 gap-3">
          <div className="form-group">
            <label className="form-label">Part Number *</label>
            <input className="form-input" value={form.part_number} onChange={e => set('part_number', e.target.value)} placeholder="SH-MRI-GC-001" />
          </div>
          <div className="form-group">
            <label className="form-label">Name *</label>
            <input className="form-input" value={form.name} onChange={e => set('name', e.target.value)} placeholder="Part name" />
          </div>
        </div>
        <div className="form-group">
          <label className="form-label">Description</label>
          <textarea className="form-input" rows={2} value={form.description || ''} onChange={e => set('description', e.target.value)} placeholder="Brief description" />
        </div>
        <div className="form-group">
          <label className="form-label">Compatible Machines <span className="font-normal text-gray-400">(comma-separated)</span></label>
          <input className="form-input" value={form.compatible_machines || ''} onChange={e => set('compatible_machines', e.target.value)} placeholder="MAGNETOM Vida, MAGNETOM Altea" />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="form-group">
            <label className="form-label">Quantity</label>
            <input type="number" min="0" className="form-input" value={form.quantity} onChange={e => set('quantity', parseInt(e.target.value) || 0)} />
          </div>
          <div className="form-group">
            <label className="form-label">Min. Quantity</label>
            <input type="number" min="1" className="form-input" value={form.min_quantity} onChange={e => set('min_quantity', parseInt(e.target.value) || 1)} />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="form-group">
            <label className="form-label">Unit Price (USD)</label>
            <input type="number" min="0" step="0.01" className="form-input" value={form.unit_price || ''} onChange={e => set('unit_price', parseFloat(e.target.value) || null)} placeholder="0.00" />
          </div>
          <div className="form-group">
            <label className="form-label">Supplier</label>
            <input className="form-input" value={form.supplier || ''} onChange={e => set('supplier', e.target.value)} placeholder="Siemens Healthineers" />
          </div>
        </div>
        <div className="flex gap-3 justify-end mt-2">
          <button className="btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={() => onSave(form)} disabled={!form.part_number || !form.name}>Save Part</button>
        </div>
      </div>
    </div>
  )
}

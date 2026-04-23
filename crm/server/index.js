const express = require('express');
const cors = require('cors');
const db = require('./db');

const app = express();
const PORT = 3001;

app.use(cors({ origin: ['http://localhost:5173', 'http://127.0.0.1:5173'] }));
app.use(express.json());

// ─── HOSPITALS ───────────────────────────────────────────────────────────────

app.get('/api/hospitals', (req, res) => {
  try {
    const hospitals = db.prepare('SELECT * FROM hospitals ORDER BY name').all();
    res.json(hospitals);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/hospitals/:id', (req, res) => {
  try {
    const hospital = db.prepare('SELECT * FROM hospitals WHERE id = ?').get(req.params.id);
    if (!hospital) return res.status(404).json({ error: 'Hospital not found' });
    res.json(hospital);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/hospitals', (req, res) => {
  try {
    const { name, city, address, phone, contact_name } = req.body;
    if (!name || !city) return res.status(400).json({ error: 'Name and city are required' });
    const stmt = db.prepare('INSERT INTO hospitals (name, city, address, phone, contact_name) VALUES (?, ?, ?, ?, ?)');
    const result = stmt.run(name, city, address || null, phone || null, contact_name || null);
    const created = db.prepare('SELECT * FROM hospitals WHERE id = ?').get(result.lastInsertRowid);
    res.status(201).json(created);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.put('/api/hospitals/:id', (req, res) => {
  try {
    const { name, city, address, phone, contact_name } = req.body;
    const stmt = db.prepare('UPDATE hospitals SET name=?, city=?, address=?, phone=?, contact_name=? WHERE id=?');
    const result = stmt.run(name, city, address || null, phone || null, contact_name || null, req.params.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Hospital not found' });
    const updated = db.prepare('SELECT * FROM hospitals WHERE id = ?').get(req.params.id);
    res.json(updated);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete('/api/hospitals/:id', (req, res) => {
  try {
    const result = db.prepare('DELETE FROM hospitals WHERE id = ?').run(req.params.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Hospital not found' });
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── MACHINES ────────────────────────────────────────────────────────────────

app.get('/api/machines', (req, res) => {
  try {
    const machines = db.prepare(`
      SELECT m.*, h.name as hospital_name, h.city as hospital_city
      FROM machines m
      JOIN hospitals h ON m.hospital_id = h.id
      ORDER BY h.name, m.type
    `).all();
    res.json(machines);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/machines/:id', (req, res) => {
  try {
    const machine = db.prepare(`
      SELECT m.*, h.name as hospital_name, h.city as hospital_city
      FROM machines m
      JOIN hospitals h ON m.hospital_id = h.id
      WHERE m.id = ?
    `).get(req.params.id);
    if (!machine) return res.status(404).json({ error: 'Machine not found' });
    res.json(machine);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/machines', (req, res) => {
  try {
    const { hospital_id, type, model, serial_number, installation_date, warranty_expiry, status, notes } = req.body;
    if (!hospital_id || !type || !model || !serial_number) {
      return res.status(400).json({ error: 'hospital_id, type, model, serial_number are required' });
    }
    const stmt = db.prepare(`
      INSERT INTO machines (hospital_id, type, model, serial_number, installation_date, warranty_expiry, status, notes)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    const result = stmt.run(hospital_id, type, model, serial_number, installation_date || null, warranty_expiry || null, status || 'active', notes || null);
    const created = db.prepare(`
      SELECT m.*, h.name as hospital_name, h.city as hospital_city
      FROM machines m JOIN hospitals h ON m.hospital_id = h.id
      WHERE m.id = ?
    `).get(result.lastInsertRowid);
    res.status(201).json(created);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.put('/api/machines/:id', (req, res) => {
  try {
    const { hospital_id, type, model, serial_number, installation_date, warranty_expiry, status, notes } = req.body;
    const stmt = db.prepare(`
      UPDATE machines SET hospital_id=?, type=?, model=?, serial_number=?, installation_date=?, warranty_expiry=?, status=?, notes=?
      WHERE id=?
    `);
    const result = stmt.run(hospital_id, type, model, serial_number, installation_date || null, warranty_expiry || null, status || 'active', notes || null, req.params.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Machine not found' });
    const updated = db.prepare(`
      SELECT m.*, h.name as hospital_name, h.city as hospital_city
      FROM machines m JOIN hospitals h ON m.hospital_id = h.id
      WHERE m.id = ?
    `).get(req.params.id);
    res.json(updated);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete('/api/machines/:id', (req, res) => {
  try {
    const result = db.prepare('DELETE FROM machines WHERE id = ?').run(req.params.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Machine not found' });
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── TICKETS ─────────────────────────────────────────────────────────────────

app.get('/api/tickets', (req, res) => {
  try {
    const tickets = db.prepare(`
      SELECT t.*, m.model as machine_model, m.type as machine_type, m.serial_number,
             h.name as hospital_name, h.city as hospital_city
      FROM tickets t
      JOIN machines m ON t.machine_id = m.id
      JOIN hospitals h ON t.hospital_id = h.id
      ORDER BY t.created_at DESC
    `).all();
    res.json(tickets);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/tickets/:id', (req, res) => {
  try {
    const ticket = db.prepare(`
      SELECT t.*, m.model as machine_model, m.type as machine_type, m.serial_number,
             h.name as hospital_name, h.city as hospital_city
      FROM tickets t
      JOIN machines m ON t.machine_id = m.id
      JOIN hospitals h ON t.hospital_id = h.id
      WHERE t.id = ?
    `).get(req.params.id);
    if (!ticket) return res.status(404).json({ error: 'Ticket not found' });
    res.json(ticket);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/tickets', (req, res) => {
  try {
    const { machine_id, hospital_id, title, description, urgency, status, assigned_to } = req.body;
    if (!machine_id || !hospital_id || !title || !urgency) {
      return res.status(400).json({ error: 'machine_id, hospital_id, title, urgency are required' });
    }
    const stmt = db.prepare(`
      INSERT INTO tickets (machine_id, hospital_id, title, description, urgency, status, assigned_to, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
    `);
    const result = stmt.run(machine_id, hospital_id, title, description || null, urgency, status || 'Open', assigned_to || null);
    const created = db.prepare(`
      SELECT t.*, m.model as machine_model, m.type as machine_type, m.serial_number,
             h.name as hospital_name, h.city as hospital_city
      FROM tickets t JOIN machines m ON t.machine_id = m.id JOIN hospitals h ON t.hospital_id = h.id
      WHERE t.id = ?
    `).get(result.lastInsertRowid);
    res.status(201).json(created);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.put('/api/tickets/:id', (req, res) => {
  try {
    const { machine_id, hospital_id, title, description, urgency, status, assigned_to, resolution_notes } = req.body;
    let resolved_at = null;
    if (status === 'Resolved' || status === 'Closed') {
      const existing = db.prepare('SELECT resolved_at FROM tickets WHERE id=?').get(req.params.id);
      resolved_at = existing && existing.resolved_at ? existing.resolved_at : new Date().toISOString();
    }
    const stmt = db.prepare(`
      UPDATE tickets SET machine_id=?, hospital_id=?, title=?, description=?, urgency=?, status=?, assigned_to=?, resolved_at=?, resolution_notes=?
      WHERE id=?
    `);
    const result = stmt.run(machine_id, hospital_id, title, description || null, urgency, status, assigned_to || null, resolved_at, resolution_notes || null, req.params.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Ticket not found' });
    const updated = db.prepare(`
      SELECT t.*, m.model as machine_model, m.type as machine_type, m.serial_number,
             h.name as hospital_name, h.city as hospital_city
      FROM tickets t JOIN machines m ON t.machine_id = m.id JOIN hospitals h ON t.hospital_id = h.id
      WHERE t.id = ?
    `).get(req.params.id);
    res.json(updated);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete('/api/tickets/:id', (req, res) => {
  try {
    const result = db.prepare('DELETE FROM tickets WHERE id = ?').run(req.params.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Ticket not found' });
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── MAINTENANCE ─────────────────────────────────────────────────────────────

app.get('/api/maintenance', (req, res) => {
  try {
    const schedules = db.prepare(`
      SELECT ms.*, m.model as machine_model, m.type as machine_type, m.serial_number,
             h.name as hospital_name, h.city as hospital_city
      FROM maintenance_schedules ms
      JOIN machines m ON ms.machine_id = m.id
      JOIN hospitals h ON ms.hospital_id = h.id
      ORDER BY ms.scheduled_date ASC
    `).all();
    res.json(schedules);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/maintenance/:id', (req, res) => {
  try {
    const schedule = db.prepare(`
      SELECT ms.*, m.model as machine_model, m.type as machine_type, m.serial_number,
             h.name as hospital_name, h.city as hospital_city
      FROM maintenance_schedules ms
      JOIN machines m ON ms.machine_id = m.id
      JOIN hospitals h ON ms.hospital_id = h.id
      WHERE ms.id = ?
    `).get(req.params.id);
    if (!schedule) return res.status(404).json({ error: 'Maintenance schedule not found' });
    res.json(schedule);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/maintenance', (req, res) => {
  try {
    const { machine_id, hospital_id, scheduled_date, type, status, notes } = req.body;
    if (!machine_id || !hospital_id || !scheduled_date || !type) {
      return res.status(400).json({ error: 'machine_id, hospital_id, scheduled_date, type are required' });
    }
    const stmt = db.prepare(`
      INSERT INTO maintenance_schedules (machine_id, hospital_id, scheduled_date, type, status, notes)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    const result = stmt.run(machine_id, hospital_id, scheduled_date, type, status || 'Scheduled', notes || null);
    const created = db.prepare(`
      SELECT ms.*, m.model as machine_model, m.type as machine_type, m.serial_number,
             h.name as hospital_name, h.city as hospital_city
      FROM maintenance_schedules ms
      JOIN machines m ON ms.machine_id = m.id JOIN hospitals h ON ms.hospital_id = h.id
      WHERE ms.id = ?
    `).get(result.lastInsertRowid);
    res.status(201).json(created);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.put('/api/maintenance/:id', (req, res) => {
  try {
    const { machine_id, hospital_id, scheduled_date, type, status, notes } = req.body;
    let completed_at = null;
    if (status === 'Completed') {
      const existing = db.prepare('SELECT completed_at FROM maintenance_schedules WHERE id=?').get(req.params.id);
      completed_at = existing && existing.completed_at ? existing.completed_at : new Date().toISOString();
    }
    const stmt = db.prepare(`
      UPDATE maintenance_schedules SET machine_id=?, hospital_id=?, scheduled_date=?, type=?, status=?, notes=?, completed_at=?
      WHERE id=?
    `);
    const result = stmt.run(machine_id, hospital_id, scheduled_date, type, status || 'Scheduled', notes || null, completed_at, req.params.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Maintenance schedule not found' });
    const updated = db.prepare(`
      SELECT ms.*, m.model as machine_model, m.type as machine_type, m.serial_number,
             h.name as hospital_name, h.city as hospital_city
      FROM maintenance_schedules ms
      JOIN machines m ON ms.machine_id = m.id JOIN hospitals h ON ms.hospital_id = h.id
      WHERE ms.id = ?
    `).get(req.params.id);
    res.json(updated);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete('/api/maintenance/:id', (req, res) => {
  try {
    const result = db.prepare('DELETE FROM maintenance_schedules WHERE id = ?').run(req.params.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Maintenance schedule not found' });
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── SPARE PARTS ─────────────────────────────────────────────────────────────

app.get('/api/parts', (req, res) => {
  try {
    const parts = db.prepare('SELECT * FROM spare_parts ORDER BY name').all();
    res.json(parts);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/parts/:id', (req, res) => {
  try {
    const part = db.prepare('SELECT * FROM spare_parts WHERE id = ?').get(req.params.id);
    if (!part) return res.status(404).json({ error: 'Part not found' });
    res.json(part);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/parts', (req, res) => {
  try {
    const { part_number, name, description, compatible_machines, quantity, min_quantity, unit_price, supplier } = req.body;
    if (!part_number || !name) return res.status(400).json({ error: 'part_number and name are required' });
    const stmt = db.prepare(`
      INSERT INTO spare_parts (part_number, name, description, compatible_machines, quantity, min_quantity, unit_price, supplier, last_updated)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    `);
    const result = stmt.run(part_number, name, description || null, compatible_machines || null, quantity || 0, min_quantity || 1, unit_price || null, supplier || null);
    const created = db.prepare('SELECT * FROM spare_parts WHERE id = ?').get(result.lastInsertRowid);
    res.status(201).json(created);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.put('/api/parts/:id', (req, res) => {
  try {
    const { part_number, name, description, compatible_machines, quantity, min_quantity, unit_price, supplier } = req.body;
    const stmt = db.prepare(`
      UPDATE spare_parts SET part_number=?, name=?, description=?, compatible_machines=?, quantity=?, min_quantity=?, unit_price=?, supplier=?, last_updated=datetime('now')
      WHERE id=?
    `);
    const result = stmt.run(part_number, name, description || null, compatible_machines || null, quantity || 0, min_quantity || 1, unit_price || null, supplier || null, req.params.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Part not found' });
    const updated = db.prepare('SELECT * FROM spare_parts WHERE id = ?').get(req.params.id);
    res.json(updated);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete('/api/parts/:id', (req, res) => {
  try {
    const result = db.prepare('DELETE FROM spare_parts WHERE id = ?').run(req.params.id);
    if (result.changes === 0) return res.status(404).json({ error: 'Part not found' });
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── DASHBOARD STATS ─────────────────────────────────────────────────────────

app.get('/api/dashboard', (req, res) => {
  try {
    const totalMachines = db.prepare("SELECT COUNT(*) as count FROM machines").get().count;
    const openTickets = db.prepare("SELECT COUNT(*) as count FROM tickets WHERE status IN ('Open','In Progress')").get().count;
    const urgentTickets = db.prepare("SELECT COUNT(*) as count FROM tickets WHERE urgency IN ('Critical','High') AND status IN ('Open','In Progress')").get().count;
    const criticalTickets = db.prepare("SELECT COUNT(*) as count FROM tickets WHERE urgency='Critical' AND status IN ('Open','In Progress')").get().count;
    const pmDue = db.prepare(`SELECT COUNT(*) as count FROM maintenance_schedules WHERE status='Scheduled' AND scheduled_date <= date('now','+30 days')`).get().count;
    const pmOverdue = db.prepare("SELECT COUNT(*) as count FROM maintenance_schedules WHERE status='Overdue'").get().count;
    const lowStock = db.prepare("SELECT COUNT(*) as count FROM spare_parts WHERE quantity <= min_quantity AND quantity > 0").get().count;
    const criticalStock = db.prepare("SELECT COUNT(*) as count FROM spare_parts WHERE quantity = 0").get().count;
    const urgentList = db.prepare(`
      SELECT t.*, m.model as machine_model, m.type as machine_type, m.serial_number,
             h.name as hospital_name, h.city as hospital_city
      FROM tickets t
      JOIN machines m ON t.machine_id = m.id
      JOIN hospitals h ON t.hospital_id = h.id
      WHERE t.urgency IN ('Critical','High') AND t.status IN ('Open','In Progress')
      ORDER BY CASE t.urgency WHEN 'Critical' THEN 0 ELSE 1 END, t.created_at DESC
      LIMIT 6
    `).all();
    const todayStr = new Date().toISOString().split('T')[0];
    const todayPM = db.prepare(`
      SELECT ms.*, m.model as machine_model, m.type as machine_type, h.name as hospital_name
      FROM maintenance_schedules ms
      JOIN machines m ON ms.machine_id = m.id
      JOIN hospitals h ON ms.hospital_id = h.id
      WHERE ms.scheduled_date = ? AND ms.status != 'Completed'
    `).all(todayStr);
    const upcomingPM = db.prepare(`
      SELECT ms.*, m.model as machine_model, m.type as machine_type, h.name as hospital_name
      FROM maintenance_schedules ms
      JOIN machines m ON ms.machine_id = m.id
      JOIN hospitals h ON ms.hospital_id = h.id
      WHERE ms.status IN ('Scheduled','Overdue') AND ms.scheduled_date <= date('now','+30 days')
      ORDER BY ms.scheduled_date ASC
      LIMIT 6
    `).all();
    res.json({ totalMachines, openTickets, urgentTickets, criticalTickets, pmDue, pmOverdue, lowStock, criticalStock, urgentList, todayPM, upcomingPM });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Al Shatta CRM Server running on http://localhost:${PORT}`);
});

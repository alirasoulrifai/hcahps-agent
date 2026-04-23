const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

const dataDir = path.join(__dirname, 'data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

const db = new Database(path.join(dataDir, 'crm.db'));

db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

function createTables() {
  db.exec(`
    CREATE TABLE IF NOT EXISTS hospitals (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      city TEXT NOT NULL,
      address TEXT,
      phone TEXT,
      contact_name TEXT,
      created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS machines (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      hospital_id INTEGER NOT NULL,
      type TEXT NOT NULL CHECK(type IN ('MRI','CT','X-Ray','Ultrasound','Angiography')),
      model TEXT NOT NULL,
      serial_number TEXT NOT NULL UNIQUE,
      installation_date TEXT,
      warranty_expiry TEXT,
      status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','inactive','under_maintenance')),
      notes TEXT,
      FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS tickets (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      machine_id INTEGER NOT NULL,
      hospital_id INTEGER NOT NULL,
      title TEXT NOT NULL,
      description TEXT,
      urgency TEXT NOT NULL CHECK(urgency IN ('Critical','High','Medium','Low')),
      status TEXT NOT NULL DEFAULT 'Open' CHECK(status IN ('Open','In Progress','Resolved','Closed')),
      assigned_to TEXT,
      created_at TEXT DEFAULT (datetime('now')),
      resolved_at TEXT,
      resolution_notes TEXT,
      FOREIGN KEY (machine_id) REFERENCES machines(id) ON DELETE CASCADE,
      FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS maintenance_schedules (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      machine_id INTEGER NOT NULL,
      hospital_id INTEGER NOT NULL,
      scheduled_date TEXT NOT NULL,
      type TEXT NOT NULL CHECK(type IN ('Bi-Annual PM','Annual PM','Corrective')),
      status TEXT NOT NULL DEFAULT 'Scheduled' CHECK(status IN ('Scheduled','Completed','Overdue')),
      notes TEXT,
      completed_at TEXT,
      FOREIGN KEY (machine_id) REFERENCES machines(id) ON DELETE CASCADE,
      FOREIGN KEY (hospital_id) REFERENCES hospitals(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS spare_parts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      part_number TEXT NOT NULL UNIQUE,
      name TEXT NOT NULL,
      description TEXT,
      compatible_machines TEXT,
      quantity INTEGER NOT NULL DEFAULT 0,
      min_quantity INTEGER NOT NULL DEFAULT 1,
      unit_price REAL,
      supplier TEXT,
      last_updated TEXT DEFAULT (datetime('now'))
    );
  `);
}

function seedData() {
  const hospitalCount = db.prepare('SELECT COUNT(*) as count FROM hospitals').get();
  if (hospitalCount.count > 0) return;

  const insertHospital = db.prepare(`
    INSERT INTO hospitals (name, city, address, phone, contact_name) VALUES (?, ?, ?, ?, ?)
  `);

  const hospitals = db.transaction(() => {
    insertHospital.run('Damascus University Hospital', 'Damascus', 'Mazzeh St, Damascus, Syria', '+963-11-3331122', 'Dr. Khaled Al-Ahmad');
    insertHospital.run('Al-Mouwasat Hospital', 'Damascus', 'Al-Mouwasat Square, Damascus, Syria', '+963-11-2224455', 'Dr. Fatima Hassan');
    insertHospital.run('University of Aleppo Hospital', 'Aleppo', 'University City, Aleppo, Syria', '+963-21-2670011', 'Dr. Omar Al-Halabi');
    insertHospital.run('Ibn Rushd Hospital', 'Aleppo', 'Al-Aziziyeh District, Aleppo, Syria', '+963-21-3345678', 'Dr. Nadia Khoury');
    insertHospital.run('Al-Bassel Hospital', 'Tartous', 'Al-Corniche Rd, Tartous, Syria', '+963-43-2201133', 'Dr. Samir Abboud');
    insertHospital.run('National Hospital Lattakia', 'Lattakia', 'Al-Hamra St, Lattakia, Syria', '+963-41-4456789', 'Dr. Rania Merhej');
    insertHospital.run('Homs National Hospital', 'Homs', 'Al-Zahraa District, Homs, Syria', '+963-31-5567890', 'Dr. Bassam Nassar');
  });
  hospitals();

  const insertMachine = db.prepare(`
    INSERT INTO machines (hospital_id, type, model, serial_number, installation_date, warranty_expiry, status, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const machines = db.transaction(() => {
    // Damascus University Hospital (id=1)
    insertMachine.run(1, 'MRI', 'MAGNETOM Vida 3T', 'MRI-SH-2021-001', '2021-03-15', '2024-03-15', 'active', 'Main MRI suite, high-field system');
    insertMachine.run(1, 'CT', 'SOMATOM Definition AS+ 128', 'CT-SH-2020-002', '2020-06-20', '2023-06-20', 'active', 'Emergency CT, 128-slice');
    insertMachine.run(1, 'X-Ray', 'YSIO Max DR', 'XR-SH-2022-003', '2022-01-10', '2025-01-10', 'active', 'Radiology department');

    // Al-Mouwasat Hospital (id=2)
    insertMachine.run(2, 'CT', 'SOMATOM go.Top', 'CT-SH-2021-004', '2021-09-05', '2024-09-05', 'under_maintenance', 'Emergency CT scanner');
    insertMachine.run(2, 'Ultrasound', 'ACUSON Sequoia', 'US-SH-2022-005', '2022-04-18', '2025-04-18', 'active', 'Cardiology unit');

    // University of Aleppo Hospital (id=3)
    insertMachine.run(3, 'MRI', 'MAGNETOM Essenza 1.5T', 'MRI-SH-2019-006', '2019-11-22', '2022-11-22', 'active', '1.5T system, general imaging');
    insertMachine.run(3, 'CT', 'SOMATOM Scope 16', 'CT-SH-2020-007', '2020-02-14', '2023-02-14', 'active', '16-slice CT');

    // Ibn Rushd Hospital (id=4)
    insertMachine.run(4, 'Angiography', 'ARTIS icono', 'ANG-SH-2022-008', '2022-07-30', '2025-07-30', 'active', 'Interventional radiology suite');
    insertMachine.run(4, 'Ultrasound', 'ACUSON Juniper', 'US-SH-2021-009', '2021-05-12', '2024-05-12', 'active', 'General ultrasound');

    // Al-Bassel Hospital (id=5)
    insertMachine.run(5, 'MRI', 'MAGNETOM Altea 1.5T', 'MRI-SH-2023-010', '2023-01-25', '2026-01-25', 'active', 'New installation, 1.5T Tim4G');
    insertMachine.run(5, 'X-Ray', 'YSIO X.pree', 'XR-SH-2021-011', '2021-08-09', '2024-08-09', 'inactive', 'Pending software upgrade');

    // National Hospital Lattakia (id=6)
    insertMachine.run(6, 'CT', 'SOMATOM go.Now', 'CT-SH-2022-012', '2022-03-20', '2025-03-20', 'active', '16-slice CT, ER access');

    // Homs National Hospital (id=7)
    insertMachine.run(7, 'MRI', 'MAGNETOM Sempra 1.5T', 'MRI-SH-2020-013', '2020-10-08', '2023-10-08', 'active', '1.5T clinical system');
    insertMachine.run(7, 'Ultrasound', 'ACUSON P500', 'US-SH-2023-014', '2023-05-17', '2026-05-17', 'active', 'Obstetrics & gynecology');
  });
  machines();

  const insertTicket = db.prepare(`
    INSERT INTO tickets (machine_id, hospital_id, title, description, urgency, status, assigned_to, created_at, resolved_at, resolution_notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const tickets = db.transaction(() => {
    insertTicket.run(4, 2, 'CT Scanner Cooling System Failure', 'SOMATOM go.Top at Al-Mouwasat reports cooling system alarm. Gantry temperature rising above threshold. Scanner offline.', 'Critical', 'In Progress', 'Ahmad Al-Zein', '2026-04-20T08:30:00', null, null);
    insertTicket.run(1, 1, 'MRI Gradient Amplifier Error', 'MAGNETOM Vida showing gradient error code G-4421. Intermittent failure during scans. Needs urgent attention.', 'High', 'Open', 'Tariq Mansour', '2026-04-21T10:15:00', null, null);
    insertTicket.run(2, 1, 'CT Image Quality Degradation', 'SOMATOM Definition AS+ producing artifacts in chest scans. Possible detector calibration issue.', 'High', 'In Progress', 'Ahmad Al-Zein', '2026-04-19T14:00:00', null, null);
    insertTicket.run(6, 3, 'MRI Table Movement Malfunction', 'MAGNETOM Essenza patient table not retracting fully. Mechanical obstruction suspected.', 'Medium', 'Open', 'Samer Khalil', '2026-04-22T09:00:00', null, null);
    insertTicket.run(11, 5, 'X-Ray Software Update Required', 'YSIO X.pree requires firmware update to v3.2.1 to resolve display glitches.', 'Low', 'Open', 'Samer Khalil', '2026-04-18T11:30:00', null, null);
    insertTicket.run(8, 4, 'Angiography Flat Detector Calibration', 'ARTIS icono flat detector requires recalibration after power fluctuation event.', 'High', 'Open', 'Tariq Mansour', '2026-04-22T13:45:00', null, null);
    insertTicket.run(13, 7, 'MRI RF Coil Connector Issue', 'MAGNETOM Sempra head coil intermittent connection. Patient exams impacted.', 'Medium', 'In Progress', 'Ahmad Al-Zein', '2026-04-17T16:20:00', null, null);
    insertTicket.run(5, 2, 'Ultrasound Probe Replacement', 'ACUSON Sequoia L18-5 linear probe showing image artifacts. Probe replacement needed.', 'Medium', 'Resolved', 'Samer Khalil', '2026-04-10T10:00:00', '2026-04-14T15:30:00', 'Replaced L18-5 linear probe. Tested and verified image quality. System fully operational.');
    insertTicket.run(3, 1, 'X-Ray Collimator Adjustment', 'YSIO Max DR collimator blades misaligned causing exposure area issues.', 'Low', 'Resolved', 'Tariq Mansour', '2026-04-05T09:30:00', '2026-04-07T11:00:00', 'Realigned collimator blades and performed calibration. All tests passed.');
    insertTicket.run(12, 6, 'CT Network Connectivity Issue', 'SOMATOM go.Now unable to push images to PACS server. Network configuration error.', 'Medium', 'Resolved', 'Ahmad Al-Zein', '2026-04-12T14:00:00', '2026-04-13T10:00:00', 'Updated DICOM network settings and restarted PACS connection. Images transferring normally.');
  });
  tickets();

  const insertMaintenance = db.prepare(`
    INSERT INTO maintenance_schedules (machine_id, hospital_id, scheduled_date, type, status, notes, completed_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `);

  const maintenance = db.transaction(() => {
    // Overdue maintenance
    insertMaintenance.run(1, 1, '2026-01-15', 'Bi-Annual PM', 'Overdue', 'Bi-annual PM for MAGNETOM Vida. Coil check, cooling system inspection.', null);
    insertMaintenance.run(6, 3, '2026-02-10', 'Annual PM', 'Overdue', 'Annual PM for MAGNETOM Essenza. Full system check required.', null);
    insertMaintenance.run(7, 3, '2026-03-05', 'Bi-Annual PM', 'Overdue', 'SOMATOM Scope 16 bi-annual service. Detector calibration.', null);

    // Upcoming maintenance
    insertMaintenance.run(2, 1, '2026-04-28', 'Bi-Annual PM', 'Scheduled', 'SOMATOM Definition AS+ semi-annual preventive maintenance.', null);
    insertMaintenance.run(4, 2, '2026-04-30', 'Corrective', 'Scheduled', 'Follow-up corrective maintenance after cooling system repair.', null);
    insertMaintenance.run(10, 5, '2026-05-05', 'Bi-Annual PM', 'Scheduled', 'MAGNETOM Altea 1.5T first bi-annual PM.', null);
    insertMaintenance.run(12, 6, '2026-05-12', 'Annual PM', 'Scheduled', 'SOMATOM go.Now annual preventive maintenance.', null);
    insertMaintenance.run(13, 7, '2026-05-20', 'Bi-Annual PM', 'Scheduled', 'MAGNETOM Sempra bi-annual PM. RF system check.', null);
    insertMaintenance.run(8, 4, '2026-06-01', 'Bi-Annual PM', 'Scheduled', 'ARTIS icono semi-annual maintenance. Flat detector check.', null);
    insertMaintenance.run(14, 7, '2026-06-15', 'Annual PM', 'Scheduled', 'ACUSON P500 annual probe and system inspection.', null);

    // Completed maintenance
    insertMaintenance.run(3, 1, '2026-03-20', 'Bi-Annual PM', 'Completed', 'YSIO Max DR bi-annual PM completed successfully.', '2026-03-20T14:00:00');
    insertMaintenance.run(5, 2, '2026-02-25', 'Annual PM', 'Completed', 'ACUSON Sequoia annual PM completed. All probes tested.', '2026-02-25T16:30:00');
    insertMaintenance.run(9, 4, '2026-03-15', 'Bi-Annual PM', 'Completed', 'ACUSON Juniper bi-annual PM. Transducer inspection done.', '2026-03-15T13:00:00');
  });
  maintenance();

  const insertPart = db.prepare(`
    INSERT INTO spare_parts (part_number, name, description, compatible_machines, quantity, min_quantity, unit_price, supplier)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const parts = db.transaction(() => {
    insertPart.run('SH-MRI-GC-001', 'MRI Gradient Coil Assembly', 'Complete gradient coil assembly for 1.5T systems', 'MAGNETOM Essenza,MAGNETOM Sempra,MAGNETOM Altea', 1, 1, 45000.00, 'Siemens Healthineers');
    insertPart.run('SH-MRI-RF-002', 'RF Head Coil 20ch', '20-channel head coil for brain imaging', 'MAGNETOM Vida,MAGNETOM Altea,MAGNETOM Sempra', 3, 2, 8500.00, 'Siemens Healthineers');
    insertPart.run('SH-MRI-CS-003', 'MRI Cooling System Compressor', 'Helium compressor unit for superconducting magnet', 'MAGNETOM Vida,MAGNETOM Essenza,MAGNETOM Sempra,MAGNETOM Altea', 0, 1, 22000.00, 'Siemens Healthineers');
    insertPart.run('SH-CT-DET-004', 'CT Detector Module 64-slice', 'Replacement detector module for CT scanners', 'SOMATOM Definition AS+,SOMATOM go.Top', 2, 2, 35000.00, 'Siemens Healthineers');
    insertPart.run('SH-CT-XRT-005', 'X-Ray Tube STRATON', 'STRATON X-ray tube for CT systems', 'SOMATOM Definition AS+,SOMATOM go.Top,SOMATOM Scope,SOMATOM go.Now', 1, 2, 28000.00, 'Siemens Healthineers');
    insertPart.run('SH-CT-HVG-006', 'High Voltage Generator Board', 'HV generator PCB for CT high voltage supply', 'SOMATOM go.Now,SOMATOM Scope', 2, 1, 12000.00, 'Siemens Healthineers');
    insertPart.run('SH-CT-COL-007', 'CT Collimator Assembly', 'Pre-patient collimator assembly complete', 'SOMATOM Definition AS+', 1, 1, 9500.00, 'Siemens Healthineers');
    insertPart.run('SH-XR-TUBE-008', 'X-Ray Tube MEGALIX', 'MEGALIX rotating anode X-ray tube', 'YSIO Max,YSIO X.pree', 2, 2, 6500.00, 'Siemens Healthineers');
    insertPart.run('SH-XR-DET-009', 'Flat Panel Detector DR', 'Amorphous silicon flat panel detector 43x43cm', 'YSIO Max,YSIO X.pree', 1, 1, 18000.00, 'Siemens Healthineers');
    insertPart.run('SH-US-PROB-010', 'Linear Probe L18-5', 'High-frequency linear transducer 5-18MHz', 'ACUSON Sequoia,ACUSON P500', 4, 2, 3200.00, 'Siemens Healthineers');
    insertPart.run('SH-US-PROB-011', 'Convex Probe C6-1', 'Broadband convex transducer 1-6MHz', 'ACUSON Sequoia,ACUSON Juniper,ACUSON P500', 3, 2, 2800.00, 'Siemens Healthineers');
    insertPart.run('SH-US-PSU-012', 'Ultrasound Power Supply Module', 'Main power supply unit for ACUSON systems', 'ACUSON Sequoia,ACUSON Juniper,ACUSON P500', 2, 1, 4500.00, 'Siemens Healthineers');
    insertPart.run('SH-ANG-FD-013', 'Angiography Flat Detector 30x30', 'CsI flat detector panel for angiography', 'ARTIS icono', 0, 1, 55000.00, 'Siemens Healthineers');
    insertPart.run('SH-ANG-TB-014', 'Angiography X-Ray Tube Megalix Cat', 'High-power rotating anode tube for angio systems', 'ARTIS icono', 1, 1, 15000.00, 'Siemens Healthineers');
    insertPart.run('SH-MRI-GA-015', 'Gradient Amplifier Module', 'Digital gradient amplifier for MRI systems', 'MAGNETOM Vida,MAGNETOM Altea', 1, 1, 32000.00, 'Siemens Healthineers');
    insertPart.run('SH-GEN-UPS-016', 'UPS Battery Module 48V', 'Replacement battery module for medical UPS systems', 'All systems', 8, 4, 1200.00, 'APC by Schneider');
    insertPart.run('SH-GEN-FILT-017', 'HEPA Air Filter Medical Grade', 'High-efficiency air filter for equipment rooms', 'All systems', 12, 6, 180.00, 'Camfil');
    insertPart.run('SH-CT-COOL-018', 'CT Cooling System Pump', 'Water cooling circulation pump for CT systems', 'SOMATOM go.Top,SOMATOM Definition AS+', 1, 1, 3800.00, 'Siemens Healthineers');
    insertPart.run('SH-MRI-CRYO-019', 'Liquid Helium Refill Kit', 'Helium refill procedure kit and consumables', 'MAGNETOM Vida,MAGNETOM Essenza,MAGNETOM Sempra,MAGNETOM Altea', 5, 3, 2500.00, 'Linde Gas Syria');
    insertPart.run('SH-GEN-CABLE-020', 'DICOM Network Cable Assembly', 'Shielded network cable set for DICOM connectivity', 'All systems', 15, 5, 120.00, 'Local Supplier');
  });
  parts();
}

createTables();
seedData();

module.exports = db;

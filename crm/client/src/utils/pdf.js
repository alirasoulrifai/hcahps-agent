import jsPDF from 'jspdf'
import autoTable from 'jspdf-autotable'

const PETROL = [0, 100, 110]
const ORANGE = [235, 120, 10]
const LIGHT_GREY = [230, 233, 235]
const DARK = [26, 26, 26]
const WHITE = [255, 255, 255]

function addHeader(doc, title) {
  doc.setFillColor(...PETROL)
  doc.rect(0, 0, 210, 28, 'F')
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(16)
  doc.setTextColor(...WHITE)
  doc.text('AL SHATTA', 14, 12)
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(8)
  doc.text('Official Distributor — Siemens Healthineers Syria', 14, 19)
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(11)
  doc.text(title, 210 - 14, 12, { align: 'right' })
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(8)
  doc.text(`Generated: ${new Date().toLocaleDateString('en-GB', { day:'2-digit', month:'short', year:'numeric' })}`, 210 - 14, 19, { align: 'right' })
}

function addFooter(doc) {
  const pageCount = doc.internal.getNumberOfPages()
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i)
    doc.setFillColor(...LIGHT_GREY)
    doc.rect(0, 285, 210, 12, 'F')
    doc.setFontSize(7)
    doc.setTextColor(100, 100, 100)
    doc.text('Al Shatta Trading Co. | Siemens Healthineers Authorized Distributor | Damascus, Syria', 14, 292)
    doc.text(`Page ${i} of ${pageCount}`, 210 - 14, 292, { align: 'right' })
  }
}

export function exportServiceReport(ticket, machine, hospital) {
  const doc = new jsPDF()
  addHeader(doc, 'SERVICE REPORT')

  doc.setFontSize(10)
  doc.setTextColor(...DARK)
  doc.setFont('helvetica', 'bold')
  doc.text('TICKET DETAILS', 14, 40)
  doc.setFont('helvetica', 'normal')

  autoTable(doc, {
    startY: 44,
    head: [],
    body: [
      ['Ticket ID', `#${ticket.id}`],
      ['Title', ticket.title],
      ['Urgency', ticket.urgency],
      ['Status', ticket.status],
      ['Hospital', `${hospital?.name || ticket.hospital_name} — ${hospital?.city || ticket.hospital_city}`],
      ['Machine', `${machine?.type || ticket.machine_type} — ${machine?.model || ticket.machine_model}`],
      ['Serial Number', machine?.serial_number || ticket.serial_number || 'N/A'],
      ['Assigned To', ticket.assigned_to || 'Unassigned'],
      ['Created', new Date(ticket.created_at).toLocaleDateString('en-GB', { day:'2-digit', month:'short', year:'numeric', hour:'2-digit', minute:'2-digit' })],
      ['Resolved', ticket.resolved_at ? new Date(ticket.resolved_at).toLocaleDateString('en-GB') : 'Pending'],
    ],
    styles: { fontSize: 9, cellPadding: 4 },
    columnStyles: { 0: { fontStyle: 'bold', fillColor: LIGHT_GREY, cellWidth: 45 } },
    headStyles: { fillColor: PETROL },
    alternateRowStyles: { fillColor: [250, 250, 250] },
  })

  const finalY = doc.lastAutoTable.finalY + 8
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(10)
  doc.text('DESCRIPTION', 14, finalY)
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(9)
  const descLines = doc.splitTextToSize(ticket.description || 'No description provided.', 182)
  doc.text(descLines, 14, finalY + 6)

  if (ticket.resolution_notes) {
    const resolveY = finalY + 6 + descLines.length * 5 + 10
    doc.setFont('helvetica', 'bold')
    doc.setFontSize(10)
    doc.text('RESOLUTION NOTES', 14, resolveY)
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(9)
    const resLines = doc.splitTextToSize(ticket.resolution_notes, 182)
    doc.text(resLines, 14, resolveY + 6)
  }

  addFooter(doc)
  doc.save(`service-report-ticket-${ticket.id}.pdf`)
}

export function exportTicketsReport(tickets) {
  const doc = new jsPDF()
  addHeader(doc, 'SERVICE TICKETS REPORT')

  autoTable(doc, {
    startY: 34,
    head: [['ID', 'Title', 'Hospital', 'Machine', 'Urgency', 'Status', 'Assigned To', 'Created']],
    body: tickets.map(t => [
      `#${t.id}`, t.title, t.hospital_name,
      `${t.machine_type} — ${t.machine_model}`,
      t.urgency, t.status, t.assigned_to || '—',
      new Date(t.created_at).toLocaleDateString('en-GB')
    ]),
    styles: { fontSize: 7.5, cellPadding: 3 },
    headStyles: { fillColor: PETROL, fontSize: 8 },
    alternateRowStyles: { fillColor: [250, 250, 250] },
    columnStyles: {
      0: { cellWidth: 10 },
      1: { cellWidth: 40 },
      4: { cellWidth: 18 },
      5: { cellWidth: 20 },
    },
    didDrawCell: (data) => {
      if (data.column.index === 4 && data.section === 'body') {
        const urgency = data.cell.raw
        if (urgency === 'Critical') doc.setTextColor(220, 38, 38)
        else if (urgency === 'High') doc.setTextColor(234, 88, 12)
        else if (urgency === 'Medium') doc.setTextColor(202, 138, 4)
        else doc.setTextColor(22, 163, 74)
      }
    },
  })

  addFooter(doc)
  doc.save(`tickets-report-${new Date().toISOString().split('T')[0]}.pdf`)
}

export function exportInventoryReport(parts) {
  const doc = new jsPDF({ orientation: 'landscape' })
  addHeader(doc, 'SPARE PARTS INVENTORY REPORT')

  autoTable(doc, {
    startY: 34,
    head: [['Part Number', 'Name', 'Compatible Machines', 'Qty', 'Min Qty', 'Status', 'Unit Price (USD)', 'Supplier']],
    body: parts.map(p => {
      const status = p.quantity === 0 ? 'OUT OF STOCK' : p.quantity <= p.min_quantity ? 'LOW STOCK' : 'OK'
      return [
        p.part_number, p.name, p.compatible_machines || '—',
        p.quantity, p.min_quantity, status,
        p.unit_price ? `$${Number(p.unit_price).toLocaleString()}` : '—',
        p.supplier || '—'
      ]
    }),
    styles: { fontSize: 7.5, cellPadding: 3 },
    headStyles: { fillColor: PETROL, fontSize: 8 },
    alternateRowStyles: { fillColor: [250, 250, 250] },
    columnStyles: {
      0: { cellWidth: 32 },
      2: { cellWidth: 55 },
      3: { cellWidth: 12, halign: 'center' },
      4: { cellWidth: 15, halign: 'center' },
      5: { cellWidth: 22, halign: 'center' },
    },
  })

  addFooter(doc)
  doc.save(`inventory-report-${new Date().toISOString().split('T')[0]}.pdf`)
}

export function exportMaintenanceReport(schedules) {
  const doc = new jsPDF()
  addHeader(doc, 'PREVENTIVE MAINTENANCE REPORT')

  autoTable(doc, {
    startY: 34,
    head: [['Hospital', 'Machine', 'Serial No.', 'PM Type', 'Scheduled Date', 'Status', 'Notes']],
    body: schedules.map(s => [
      s.hospital_name, `${s.machine_type} — ${s.machine_model}`,
      s.serial_number, s.type,
      new Date(s.scheduled_date).toLocaleDateString('en-GB'),
      s.status, s.notes || '—'
    ]),
    styles: { fontSize: 7.5, cellPadding: 3 },
    headStyles: { fillColor: PETROL, fontSize: 8 },
    alternateRowStyles: { fillColor: [250, 250, 250] },
    columnStyles: {
      0: { cellWidth: 35 },
      1: { cellWidth: 38 },
      2: { cellWidth: 28 },
    },
  })

  addFooter(doc)
  doc.save(`pm-report-${new Date().toISOString().split('T')[0]}.pdf`)
}

export function exportInstalledBaseReport(hospitals, machines) {
  const doc = new jsPDF({ orientation: 'landscape' })
  addHeader(doc, 'INSTALLED BASE REPORT')

  autoTable(doc, {
    startY: 34,
    head: [['Hospital', 'City', 'Machine Type', 'Model', 'Serial Number', 'Installation Date', 'Warranty Expiry', 'Status']],
    body: machines.map(m => {
      const h = hospitals.find(h => h.id === m.hospital_id)
      const today = new Date()
      const wExp = m.warranty_expiry ? new Date(m.warranty_expiry) : null
      const wStatus = !wExp ? 'N/A' : wExp > today ? 'Active' : 'Expired'
      return [
        m.hospital_name || h?.name || '—', m.hospital_city || h?.city || '—',
        m.type, m.model, m.serial_number,
        m.installation_date ? new Date(m.installation_date).toLocaleDateString('en-GB') : '—',
        wExp ? wExp.toLocaleDateString('en-GB') : '—',
        m.status
      ]
    }),
    styles: { fontSize: 7.5, cellPadding: 3 },
    headStyles: { fillColor: PETROL, fontSize: 8 },
    alternateRowStyles: { fillColor: [250, 250, 250] },
    columnStyles: {
      0: { cellWidth: 40 },
      3: { cellWidth: 38 },
    },
  })

  addFooter(doc)
  doc.save(`installed-base-${new Date().toISOString().split('T')[0]}.pdf`)
}

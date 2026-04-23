import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './components/Dashboard'
import InstalledBase from './components/InstalledBase'
import Tickets from './components/Tickets'
import PMCalendar from './components/PMCalendar'
import SpareParts from './components/SpareParts'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="installed-base" element={<InstalledBase />} />
          <Route path="tickets" element={<Tickets />} />
          <Route path="maintenance" element={<PMCalendar />} />
          <Route path="spare-parts" element={<SpareParts />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

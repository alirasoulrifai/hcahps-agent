import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const hospitalsApi = {
  getAll: () => api.get('/hospitals').then(r => r.data),
  getById: id => api.get(`/hospitals/${id}`).then(r => r.data),
  getMachines: id => api.get(`/hospitals/${id}/machines`).then(r => r.data),
  create: data => api.post('/hospitals', data).then(r => r.data),
  update: (id, data) => api.put(`/hospitals/${id}`, data).then(r => r.data),
  remove: id => api.delete(`/hospitals/${id}`),
}

export const machinesApi = {
  getAll: () => api.get('/machines').then(r => r.data),
  getById: id => api.get(`/machines/${id}`).then(r => r.data),
  create: data => api.post('/machines', data).then(r => r.data),
  update: (id, data) => api.put(`/machines/${id}`, data).then(r => r.data),
  remove: id => api.delete(`/machines/${id}`),
}

export const ticketsApi = {
  getAll: () => api.get('/tickets').then(r => r.data),
  create: data => api.post('/tickets', data).then(r => r.data),
  update: (id, data) => api.put(`/tickets/${id}`, data).then(r => r.data),
  remove: id => api.delete(`/tickets/${id}`),
}

export const maintenanceApi = {
  getAll: () => api.get('/maintenance').then(r => r.data),
  create: data => api.post('/maintenance', data).then(r => r.data),
  update: (id, data) => api.put(`/maintenance/${id}`, data).then(r => r.data),
  remove: id => api.delete(`/maintenance/${id}`),
}

export const partsApi = {
  getAll: () => api.get('/parts').then(r => r.data),
  create: data => api.post('/parts', data).then(r => r.data),
  update: (id, data) => api.put(`/parts/${id}`, data).then(r => r.data),
  remove: id => api.delete(`/parts/${id}`),
}

export const dashboardApi = {
  get: () => api.get('/dashboard').then(r => r.data),
}

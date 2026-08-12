import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Alerts from './pages/Alerts'
import Dashboard from './pages/Dashboard'
import History from './pages/History'
import Login from './pages/Login'
import Speakers from './pages/Speakers'
import SpeakerWizard from './pages/SpeakerWizard'
import './App.css'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/history" element={<History />} />
          <Route path="/speakers" element={<Speakers />} />
          <Route path="/speakers/new" element={<SpeakerWizard />} />
          <Route path="/alerts" element={<Alerts />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

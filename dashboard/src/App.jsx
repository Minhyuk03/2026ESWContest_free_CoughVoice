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
          {/* 경로가 /alerts면 서버의 GET /alerts API와 겹쳐 화면이 안 열린다.
              서버가 대시보드를 직접 서빙하므로 API 라우트가 먼저 잡힌다. */}
          <Route path="/alert-center" element={<Alerts />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

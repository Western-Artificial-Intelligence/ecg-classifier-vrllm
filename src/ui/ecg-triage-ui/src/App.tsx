import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import NewLandingPage from './pages/NewLandingPage';
import PremiumLandingPage from './pages/PremiumLandingPage';
import LandingPage from './pages/LandingPage';
import PatientDashboard from './pages/PatientDashboard';
import PatientAnalysis from './pages/PatientAnalysis';
import AppMainMenu from './pages/AppMainMenu';
import PatientInfoPlaceholder from './pages/PatientInfoPlaceholder';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<NewLandingPage />} />
        <Route path="/premium" element={<PremiumLandingPage />} />
        <Route path="/app" element={<AppMainMenu />} />
        <Route path="/app/patient" element={<PatientInfoPlaceholder />} />
        <Route path="/app/analysis" element={<PatientAnalysis />} />
        <Route path="/old" element={<LandingPage />} />
        <Route path="/dashboard" element={<PatientDashboard />} />
        <Route path="/analysis" element={<PatientAnalysis />} />
        <Route path="/analysis/:patientId" element={<PatientAnalysis />} />
      </Routes>
    </Router>
  );
};

export default App;

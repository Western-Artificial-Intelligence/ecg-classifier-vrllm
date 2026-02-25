import React from 'react';
import { Routes, Route } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import MainMenu from './components/MainMenu';
import SelectPatients from './components/SelectPatients';
import PatientAnalysis from './components/PatientAnalysis';
import PatientManagement from './components/PatientManagement';

const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/menu" element={<MainMenu />} />
      <Route path="/patients" element={<SelectPatients />} />
      <Route path="/patient-management" element={<PatientManagement />} />
      <Route path="/patient-management/:patientId" element={<PatientManagement />} />
      <Route path="/analysis" element={<PatientAnalysis />} />
    </Routes>
  );
};

export default App;

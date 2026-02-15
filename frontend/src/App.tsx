import React from 'react';
import { Routes, Route } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import PatientAnalysis from './components/PatientAnalysis';

const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/analysis" element={<PatientAnalysis />} />
    </Routes>
  );
};

export default App;
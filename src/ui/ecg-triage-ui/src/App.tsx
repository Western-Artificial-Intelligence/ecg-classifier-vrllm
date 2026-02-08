import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NewLandingPage from './pages/NewLandingPage';
import LandingPage from './pages/LandingPage';
import PatientAnalysis from './pages/PatientAnalysis';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<NewLandingPage />} />
        <Route path="/old" element={<LandingPage />} />
        <Route path="/analysis" element={<PatientAnalysis />} />
      </Routes>
    </Router>
  );
};

export default App;

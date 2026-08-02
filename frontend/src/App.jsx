import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import DashboardView from './components/DashboardView';
import DataCollectView from './components/DataCollectView';
import AnnotateView from './components/AnnotateView';
import DatasetView from './components/DatasetView';
import TrainView from './components/TrainView';

export default function App() {
  const [activeTab, setActiveTab] = useState(() => {
    return localStorage.getItem('labelin_active_tab') || 'dashboard';
  });

  useEffect(() => {
    localStorage.setItem('labelin_active_tab', activeTab);
  }, [activeTab]);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', width: '100vw', backgroundColor: 'var(--bg-main)' }}>
      {/* Roboflow Dark Left Navigation Bar */}
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Workspace Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Header activeTab={activeTab} />
        <main style={{ flex: 1, overflowY: 'auto', position: 'relative' }}>
          {/* Keep all tab views mounted in DOM to prevent progress reset & input loss */}
          <div style={{ display: activeTab === 'dashboard' ? 'block' : 'none' }}>
            <DashboardView setActiveTab={setActiveTab} />
          </div>

          <div style={{ display: activeTab === 'collect' ? 'block' : 'none' }}>
            <DataCollectView setActiveTab={setActiveTab} />
          </div>

          <div style={{ display: activeTab === 'annotate' ? 'block' : 'none' }}>
            <AnnotateView activeTab={activeTab} />
          </div>

          <div style={{ display: activeTab === 'dataset' ? 'block' : 'none' }}>
            <DatasetView setActiveTab={setActiveTab} />
          </div>

          <div style={{ display: activeTab === 'train' ? 'block' : 'none' }}>
            <TrainView />
          </div>
        </main>
      </div>
    </div>
  );
}

import React from 'react';
import { Activity, ShieldCheck } from 'lucide-react';

export default function Header({ activeTab }) {
  const titleMap = {
    dashboard: 'Workspace Overview & Metrics',
    collect: 'Data Acquisition & Live Stream Capture',
    annotate: 'Visual Labeling Workspace & AI Assistant',
    dataset: 'Dataset Splitting & Export Pipeline',
    train: 'YOLO Model Training Hub & Real-time Logs'
  };

  return (
    <header style={{
      height: '64px',
      borderBottom: '1px solid var(--border-color)',
      backgroundColor: 'rgba(15, 19, 34, 0.75)',
      backdropFilter: 'blur(10px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 24px',
      position: 'sticky',
      top: 0,
      zIndex: 10
    }}>
      <div>
        <h2 style={{ fontSize: '1.15rem', fontWeight: 600, color: '#FFF' }}>
          {titleMap[activeTab] || 'Workspace'}
        </h2>
        <p style={{ fontSize: '0.78rem', color: 'var(--text-subtle)' }}>
          Project: Traffic Detection Suite • Location: dataset_raw / dataset_manual
        </p>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '4px 10px',
          borderRadius: '9999px',
          background: 'rgba(16, 185, 129, 0.1)',
          border: '1px solid rgba(16, 185, 129, 0.3)',
          color: 'var(--accent-success)',
          fontSize: '0.78rem',
          fontWeight: 600
        }}>
          <Activity size={14} />
          FastAPI Backend Online
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '4px 10px',
          borderRadius: '9999px',
          background: 'rgba(99, 102, 241, 0.1)',
          border: '1px solid rgba(99, 102, 241, 0.3)',
          color: 'var(--accent-primary)',
          fontSize: '0.78rem',
          fontWeight: 600
        }}>
          <ShieldCheck size={14} />
          YOLO Traffic Classes (4)
        </div>
      </div>
    </header>
  );
}

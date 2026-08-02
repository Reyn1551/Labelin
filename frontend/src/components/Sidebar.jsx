import React from 'react';
import { LayoutDashboard, Video, PenTool, Database, Rocket, Layers } from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'collect', label: 'Data Collect', icon: Video },
    { id: 'annotate', label: 'Annotate Canvas', icon: PenTool },
    { id: 'dataset', label: 'Dataset Split', icon: Database },
    { id: 'train', label: 'YOLO Training', icon: Rocket },
  ];

  return (
    <aside style={{
      width: '240px',
      backgroundColor: 'var(--bg-sidebar)',
      borderRight: '1px solid var(--border-color)',
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      flexShrink: 0
    }}>
      {/* Brand Header */}
      <div style={{
        padding: '20px 16px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        borderBottom: '1px solid var(--border-color)'
      }}>
        <div style={{
          width: '36px',
          height: '36px',
          borderRadius: '10px',
          background: 'linear-gradient(135deg, #6366F1, #8B5CF6)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 15px rgba(99, 102, 241, 0.4)'
        }}>
          <Layers size={20} color="#FFFFFF" />
        </div>
        <div>
          <h1 style={{ fontSize: '1.1rem', fontWeight: 700, letterSpacing: '-0.02em', color: '#FFF' }}>Labelin</h1>
          <span style={{ fontSize: '0.72rem', color: 'var(--accent-purple)', fontWeight: 600, letterSpacing: '0.05em' }}>ROBOFLOW UI EDITION</span>
        </div>
      </div>

      {/* Navigation Links */}
      <nav style={{ padding: '16px 8px', flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <div style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-subtle)', padding: '0 12px 8px 12px', letterSpacing: '0.05em' }}>
          WORKSPACE
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '10px 12px',
                borderRadius: '8px',
                border: 'none',
                background: isActive ? 'linear-gradient(90deg, rgba(99, 102, 241, 0.18), rgba(139, 92, 246, 0.08))' : 'transparent',
                color: isActive ? '#FFFFFF' : 'var(--text-muted)',
                fontWeight: isActive ? 600 : 400,
                fontSize: '0.9rem',
                cursor: 'pointer',
                textAlign: 'left',
                borderLeft: isActive ? '3px solid var(--accent-primary)' : '3px solid transparent',
                transition: 'all 0.15s ease'
              }}
            >
              <Icon size={18} color={isActive ? 'var(--accent-primary)' : 'var(--text-muted)'} />
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div style={{
        padding: '16px',
        borderTop: '1px solid var(--border-color)',
        fontSize: '0.75rem',
        color: 'var(--text-subtle)'
      }}>
        <div>System: <span style={{ color: 'var(--accent-success)', fontWeight: 600 }}>Web Active</span></div>
        <div>Model: YOLOv8 / YOLOv11</div>
      </div>
    </aside>
  );
}

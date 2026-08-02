import React, { useState, useEffect } from 'react';
import { Images, CheckCircle, FolderOpen, Zap, ArrowRight, Play, Database, RefreshCw } from 'lucide-react';

export default function DashboardView({ setActiveTab }) {
  const [stats, setStats] = useState({
    rawImages: 0,
    labeledImages: 0,
    datasetImages: 0,
    loading: true,
    error: false
  });

  useEffect(() => {
    fetchStats();
    const interval = setInterval(() => {
      fetchStats();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const rawRes = await fetch('http://localhost:8000/api/annotate/images?folder=dataset_raw');
      const rawData = await rawRes.json();
      
      const manualRes = await fetch('http://localhost:8000/api/annotate/images?folder=dataset_manual');
      const manualData = await manualRes.json();

      setStats({
        rawImages: rawData.total || 0,
        labeledImages: manualData.total || 0,
        datasetImages: (manualData.total || 0),
        loading: false,
        error: false
      });
    } catch (e) {
      console.error('Error fetching dashboard stats:', e);
      setStats((prev) => ({ ...prev, loading: false, error: true }));
    }
  };

  return (
    <div style={{ padding: '28px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Top Banner */}
      <div className="rf-card rf-card-glow" style={{
        padding: '24px',
        background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.05))',
        borderColor: 'rgba(99, 102, 241, 0.4)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-purple)', letterSpacing: '0.05em' }}>ROBOFLOW WORKSPACE</span>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '4px', color: '#FFF' }}>Traffic Object Detection Pipeline</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '6px', maxWidth: '600px' }}>
            Capture live traffic streams (M3U8 / RTSP), auto-label frames using YOLO models, refine annotations on an interactive canvas, and train custom YOLO models.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button className="rf-btn rf-btn-secondary" onClick={fetchStats} title="Refresh Stats">
            <RefreshCw size={16} /> Sync Data
          </button>
          <button className="rf-btn rf-btn-primary" onClick={() => setActiveTab('collect')} style={{ padding: '12px 20px', fontSize: '0.95rem' }}>
            <Play size={18} /> Launch Data Capture
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px' }}>
        <div className="rf-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 500 }}>Raw Frame Dataset</span>
            <div style={{ padding: '8px', borderRadius: '8px', background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)' }}>
              <Images size={20} />
            </div>
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 700, marginTop: '12px', color: '#FFF' }}>
            {stats.loading ? '...' : stats.rawImages}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>Directory: dataset_raw/</span>
        </div>

        <div className="rf-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 500 }}>Manual / Auto Labeled</span>
            <div style={{ padding: '8px', borderRadius: '8px', background: 'rgba(16, 185, 129, 0.15)', color: 'var(--accent-success)' }}>
              <CheckCircle size={20} />
            </div>
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 700, marginTop: '12px', color: '#FFF' }}>
            {stats.loading ? '...' : stats.labeledImages}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>Directory: dataset_manual/</span>
        </div>

        <div className="rf-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 500 }}>Target Classes</span>
            <div style={{ padding: '8px', borderRadius: '8px', background: 'rgba(139, 92, 246, 0.15)', color: 'var(--accent-purple)' }}>
              <Zap size={20} />
            </div>
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 700, marginTop: '12px', color: '#FFF' }}>4</div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>Car, Motorcycle, Bus, Truck</span>
        </div>

        <div className="rf-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 500 }}>Train / Val / Test Export</span>
            <div style={{ padding: '8px', borderRadius: '8px', background: 'rgba(245, 158, 11, 0.15)', color: 'var(--accent-warning)' }}>
              <Database size={20} />
            </div>
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 700, marginTop: '12px', color: '#FFF' }}>YOLO YAML</div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>File: dataset/traffic.yaml</span>
        </div>
      </div>

      {/* Quick Action Navigation Grid */}
      <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#FFF', marginTop: '8px' }}>Pipeline Workflow Steps</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
        <div className="rf-card" style={{ padding: '20px', cursor: 'pointer' }} onClick={() => setActiveTab('collect')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
            <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'var(--accent-primary)', color: '#FFF', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '0.85rem' }}>1</div>
            <h4 style={{ fontWeight: 600, color: '#FFF' }}>Data Acquisition</h4>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Input RTSP or M3U8 traffic stream URLs, extract frames with custom frame-skip intervals.</p>
          <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-primary)', fontSize: '0.85rem', fontWeight: 600 }}>
            Open Stream Capture <ArrowRight size={16} />
          </div>
        </div>

        <div className="rf-card" style={{ padding: '20px', cursor: 'pointer' }} onClick={() => setActiveTab('annotate')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
            <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'var(--accent-purple)', color: '#FFF', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '0.85rem' }}>2</div>
            <h4 style={{ fontWeight: 600, color: '#FFF' }}>Annotation Canvas</h4>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Draw, resize, and edit bounding boxes on canvas. Auto-label using YOLO AI assistant.</p>
          <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-purple)', fontSize: '0.85rem', fontWeight: 600 }}>
            Open Editor <ArrowRight size={16} />
          </div>
        </div>

        <div className="rf-card" style={{ padding: '20px', cursor: 'pointer' }} onClick={() => setActiveTab('dataset')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
            <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'var(--accent-warning)', color: '#FFF', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '0.85rem' }}>3</div>
            <h4 style={{ fontWeight: 600, color: '#FFF' }}>Dataset Prep</h4>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Split images into train/val/test folders, export standard Ultralytics traffic.yaml file.</p>
          <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-warning)', fontSize: '0.85rem', fontWeight: 600 }}>
            Configure Split <ArrowRight size={16} />
          </div>
        </div>

        <div className="rf-card" style={{ padding: '20px', cursor: 'pointer' }} onClick={() => setActiveTab('train')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
            <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'var(--accent-success)', color: '#FFF', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '0.85rem' }}>4</div>
            <h4 style={{ fontWeight: 600, color: '#FFF' }}>Model Training</h4>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Train custom YOLO model with live streaming terminal logs and epoch metric updates.</p>
          <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-success)', fontSize: '0.85rem', fontWeight: 600 }}>
            Launch Training <ArrowRight size={16} />
          </div>
        </div>
      </div>
    </div>
  );
}

import React, { useState, useEffect, useRef } from 'react';
import { Rocket, Cpu, Terminal, Play, CheckCircle2, AlertTriangle, RefreshCw } from 'lucide-react';

export default function TrainView() {
  const [yamlPath, setYamlPath] = useState('dataset/traffic.yaml');
  const [modelPath, setModelPath] = useState('yolo11n.pt');
  const [epochs, setEpochs] = useState(50);
  const [batch, setBatch] = useState(16);
  const [imgsz, setImgsz] = useState(640);
  const [workers, setWorkers] = useState(2);

  const [logs, setLogs] = useState([]);
  const [trainingStatus, setTrainingStatus] = useState({ is_running: false, metrics: {} });
  const wsRef = useRef(null);
  const terminalEndRef = useRef(null);

  useEffect(() => {
    fetchStatus();
    connectWebSocket();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const fetchStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/train/status');
      const data = await res.json();
      setTrainingStatus(data);
      if (data.recent_logs && data.recent_logs.length > 0) {
        setLogs(data.recent_logs);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8000/api/train/ws/logs');
    ws.onmessage = (event) => {
      setLogs((prev) => [...prev, event.data]);
    };
    ws.onerror = (e) => console.error('WS error:', e);
    wsRef.current = ws;
  };

  const handleStartTraining = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          yaml_path: yamlPath,
          model_path: modelPath,
          epochs: parseInt(epochs),
          batch: parseInt(batch),
          imgsz: parseInt(imgsz),
          workers: parseInt(workers)
        })
      });
      const data = await res.json();
      if (!res.ok) {
        alert(data.detail || 'Failed to start training');
      } else {
        fetchStatus();
      }
    } catch (e) {
      alert('Error initiating training: ' + e.message);
    }
  };

  const metrics = trainingStatus.metrics || {};

  return (
    <div style={{ padding: '28px', display: 'grid', gridTemplateColumns: '340px 1fr', gap: '24px' }}>
      {/* Left Column: Hyperparameters & Configuration */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <div className="rf-card" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#FFF', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <Rocket size={20} color="var(--accent-success)" /> YOLO Training Hyperparameters
          </h3>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Traffic YAML Config Path</label>
            <input type="text" className="rf-input" value={yamlPath} onChange={(e) => setYamlPath(e.target.value)} />
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Base Weights Model</label>
            <input type="text" className="rf-input" value={modelPath} onChange={(e) => setModelPath(e.target.value)} />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
            <div>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Epochs</label>
              <input type="number" className="rf-input" value={epochs} onChange={(e) => setEpochs(e.target.value)} min={1} max={500} />
            </div>
            <div>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Batch Size</label>
              <input type="number" className="rf-input" value={batch} onChange={(e) => setBatch(e.target.value)} min={1} max={128} />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '20px' }}>
            <div>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Image Size (px)</label>
              <input type="number" className="rf-input" value={imgsz} onChange={(e) => setImgsz(e.target.value)} step={32} />
            </div>
            <div>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Workers</label>
              <input type="number" className="rf-input" value={workers} onChange={(e) => setWorkers(e.target.value)} min={0} max={8} />
            </div>
          </div>

          <button
            className="rf-btn rf-btn-primary"
            style={{ width: '100%', padding: '12px', background: 'linear-gradient(135deg, #10B981, #059669)' }}
            onClick={handleStartTraining}
            disabled={trainingStatus.is_running}
          >
            <Play size={18} /> {trainingStatus.is_running ? 'Training in Progress...' : 'Start YOLO Training'}
          </button>
        </div>

        {/* Training Metrics Card */}
        <div className="rf-card" style={{ padding: '24px' }}>
          <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#FFF', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <Cpu size={18} color="var(--accent-purple)" /> Real-Time Metrics & Status
          </h4>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Status:</span>
              <span style={{ color: trainingStatus.is_running ? 'var(--accent-warning)' : 'var(--accent-success)', fontWeight: 600 }}>
                {metrics.status || (trainingStatus.is_running ? 'training' : 'idle')}
              </span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Epoch:</span>
              <span style={{ color: '#FFF', fontWeight: 600 }}>
                {metrics.epoch || 0} / {metrics.total_epochs || epochs}
              </span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Loss:</span>
              <span style={{ color: 'var(--accent-cyan)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                {metrics.loss ? metrics.loss.toFixed(4) : '0.0000'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Column: WebSocket Live Terminal */}
      <div className="rf-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', height: 'calc(100vh - 120px)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#FFF', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Terminal size={20} color="var(--accent-primary)" /> Live YOLO Training Terminal (WebSocket)
          </h3>
          <button className="rf-btn rf-btn-secondary" style={{ padding: '4px 8px' }} onClick={() => setLogs([])}>
            <RefreshCw size={14} /> Clear Logs
          </button>
        </div>

        <div className="rf-terminal" style={{ flex: 1 }}>
          {logs.map((line, idx) => (
            <div key={idx} style={{ color: line.includes('Epoch') ? '#A7F3D0' : line.includes('Error') ? '#FCA5A5' : '#38BDF8' }}>
              {line}
            </div>
          ))}
          {logs.length === 0 && (
            <div style={{ color: 'var(--text-subtle)' }}>
              WebSocket connected. Click 'Start YOLO Training' to begin live log streaming.
            </div>
          )}
          <div ref={terminalEndRef} />
        </div>
      </div>
    </div>
  );
}

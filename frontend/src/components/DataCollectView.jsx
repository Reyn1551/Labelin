import React, { useState, useEffect } from 'react';
import { Video, Play, StopCircle, Plus, Trash2, Wand2, RefreshCw, UploadCloud, CheckCircle, Sliders, Zap } from 'lucide-react';

export default function DataCollectView({ setActiveTab }) {
  const [sources, setSources] = useState(() => {
    try {
      const saved = localStorage.getItem('labelin_sources');
      return saved ? JSON.parse(saved) : ['https://cctv.jogjakota.go.id/malioboro/Malioboro_4_Kepatihan.stream/chunklist_w12345.m3u8'];
    } catch {
      return ['https://cctv.jogjakota.go.id/malioboro/Malioboro_4_Kepatihan.stream/chunklist_w12345.m3u8'];
    }
  });

  const [newSource, setNewSource] = useState('');

  const [numFrames, setNumFrames] = useState(() => {
    return localStorage.getItem('labelin_num_frames') || 200;
  });

  const [frameSkip, setFrameSkip] = useState(() => {
    return localStorage.getItem('labelin_frame_skip') || 2;
  });

  // Model selection & upload state
  const [availableModels, setAvailableModels] = useState(['yolov8n.pt', 'yolov8x.pt', 'yolo11x.pt']);

  const [modelPath, setModelPath] = useState(() => {
    return localStorage.getItem('labelin_model_path') || 'yolov8x.pt';
  });

  const [autolabelConf, setAutolabelConf] = useState(() => {
    return localStorage.getItem('labelin_autolabel_conf') || 0.20;
  });

  const [autolabelIou, setAutolabelIou] = useState(() => {
    return localStorage.getItem('labelin_autolabel_iou') || 0.40;
  });

  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('');

  const [captureStatus, setCaptureStatus] = useState({ is_running: false, progress: 0, total: 0, logs: [] });
  const [autolabelStatus, setAutolabelStatus] = useState({ is_running: false, progress: 0, total: 0, logs: [] });

  // Sync to localStorage
  useEffect(() => {
    localStorage.setItem('labelin_sources', JSON.stringify(sources));
    saveSourcesToBackend(sources);
  }, [sources]);

  useEffect(() => {
    localStorage.setItem('labelin_num_frames', numFrames);
  }, [numFrames]);

  useEffect(() => {
    localStorage.setItem('labelin_frame_skip', frameSkip);
  }, [frameSkip]);

  useEffect(() => {
    localStorage.setItem('labelin_model_path', modelPath);
  }, [modelPath]);

  useEffect(() => {
    localStorage.setItem('labelin_autolabel_conf', autolabelConf);
  }, [autolabelConf]);

  useEffect(() => {
    localStorage.setItem('labelin_autolabel_iou', autolabelIou);
  }, [autolabelIou]);

  useEffect(() => {
    fetchBackendSources();
    fetchModels();
    fetchCaptureStatus();
    fetchAutolabelStatus();

    const interval = setInterval(() => {
      fetchCaptureStatus();
      fetchAutolabelStatus();
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  const fetchBackendSources = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/capture/sources');
      const data = await res.json();
      if (data.sources && data.sources.length > 0) {
        setSources(data.sources);
      }
    } catch (e) {
      console.error('Error loading backend sources:', e);
    }
  };

  const saveSourcesToBackend = async (sourcesList) => {
    try {
      await fetch('http://localhost:8000/api/capture/sources', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sources: sourcesList })
      });
    } catch (e) {
      console.error('Error saving sources to backend:', e);
    }
  };

  const fetchModels = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/annotate/models');
      const data = await res.json();
      if (data.models && data.models.length > 0) {
        setAvailableModels(data.models);
      }
    } catch (e) {
      console.error('Error loading models:', e);
    }
  };

  const fetchCaptureStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/capture/status');
      const data = await res.json();
      setCaptureStatus(data);
    } catch (e) {
      console.error('Error loading capture status:', e);
    }
  };

  const fetchAutolabelStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/annotate/autolabel/status');
      const data = await res.json();
      setAutolabelStatus(data);
    } catch (e) {
      console.error('Error loading autolabel status:', e);
    }
  };

  const addSource = () => {
    if (!newSource.trim()) return;
    setSources([...sources, newSource.trim()]);
    setNewSource('');
  };

  const removeSource = (index) => {
    setSources(sources.filter((_, i) => i !== index));
  };

  const handleStartCapture = async () => {
    try {
      await fetch('http://localhost:8000/api/capture/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sources: sources,
          num_frames: parseInt(numFrames) || 200,
          frame_skip: parseInt(frameSkip) || 2
        })
      });
      fetchCaptureStatus();
    } catch (e) {
      console.error('Error starting capture:', e);
    }
  };

  const handleCancelCapture = async () => {
    try {
      await fetch('http://localhost:8000/api/capture/cancel', { method: 'POST' });
      fetchCaptureStatus();
    } catch (e) {
      console.error('Error cancelling capture:', e);
    }
  };

  const handleStartAutolabel = async () => {
    try {
      await fetch('http://localhost:8000/api/annotate/autolabel/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_path: modelPath,
          image_dir: 'dataset_raw',
          output_dir: 'dataset_labeled',
          conf: parseFloat(autolabelConf),
          iou: parseFloat(autolabelIou)
        })
      });
      fetchAutolabelStatus();
    } catch (e) {
      console.error('Error starting autolabel:', e);
    }
  };

  const handleCancelAutolabel = async () => {
    try {
      await fetch('http://localhost:8000/api/annotate/autolabel/cancel', { method: 'POST' });
      fetchAutolabelStatus();
    } catch (e) {
      console.error('Error cancelling autolabel:', e);
    }
  };

  const handleModelFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setUploading(true);
    setUploadMsg('Uploading model...');

    try {
      const res = await fetch('http://localhost:8000/api/annotate/models/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        setUploadMsg(`Uploaded ${data.filename}!`);
        await fetchModels();
        setModelPath(data.model_path);
      } else {
        setUploadMsg(`Error: ${data.detail}`);
      }
    } catch (err) {
      setUploadMsg('Failed to upload model.');
    } finally {
      setUploading(false);
    }
  };

  const applyOptimalPreset = () => {
    setAutolabelConf(0.20);
    setAutolabelIou(0.40);
  };

  const allLogs = [
    ...(captureStatus.logs || []).map(l => `[CAPTURE] ${l}`),
    ...(autolabelStatus.logs || []).map(l => `[AUTOLABEL] ${l}`)
  ];

  return (
    <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Title & Description */}
      <div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#FFF', marginBottom: '6px' }}>
          Step 1: Data Collection & AI Auto-Labeling
        </h2>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
          Capture live traffic frames from CCTV video streams (M3U8 / RTSP) and auto-annotate them with high-accuracy YOLO models.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        {/* Left Column: Stream Capture Settings */}
        <div className="rf-card" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#FFF', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <Video size={20} color="var(--accent-cyan)" /> CCTV Video Stream Sources
          </h3>

          {/* Add Stream URL */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
            <input
              type="text"
              className="rf-input"
              placeholder="Paste stream URL (M3U8 / RTSP / Video file path)..."
              value={newSource}
              onChange={(e) => setNewSource(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && addSource()}
            />
            <button className="rf-btn rf-btn-secondary" onClick={addSource}>
              <Plus size={16} /> Add
            </button>
          </div>

          {/* Stream List */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '20px', maxHeight: '160px', overflowY: 'auto' }}>
            {sources.map((src, idx) => (
              <div key={idx} style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '8px 12px',
                background: 'var(--bg-input)',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
                fontSize: '0.85rem'
              }}>
                <span style={{ color: 'var(--text-main)', wordBreak: 'break-all', fontFamily: 'var(--font-mono)' }}>{src}</span>
                <button style={{ background: 'transparent', border: 'none', color: 'var(--accent-danger)', cursor: 'pointer' }} onClick={() => removeSource(idx)}>
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </div>

          {/* Controls */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
            <div>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Frames to Capture</label>
              <input type="number" className="rf-input" value={numFrames} onChange={(e) => setNumFrames(e.target.value)} min={10} max={5000} />
            </div>
            <div>
              <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Frame Skip Interval</label>
              <input type="number" className="rf-input" value={frameSkip} onChange={(e) => setFrameSkip(e.target.value)} min={1} max={30} />
            </div>
          </div>

          {/* Presets */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '20px' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-subtle)', alignSelf: 'center' }}>Presets:</span>
            <button className="rf-btn rf-btn-secondary" style={{ padding: '4px 10px', fontSize: '0.75rem' }} onClick={() => setNumFrames(50)}>50 frames</button>
            <button className="rf-btn rf-btn-secondary" style={{ padding: '4px 10px', fontSize: '0.75rem' }} onClick={() => setNumFrames(200)}>200 frames</button>
            <button className="rf-btn rf-btn-secondary" style={{ padding: '4px 10px', fontSize: '0.75rem' }} onClick={() => setNumFrames(500)}>500 frames</button>
          </div>

          {/* Action Button */}
          {captureStatus.is_running ? (
            <button className="rf-btn rf-btn-danger" style={{ width: '100%' }} onClick={handleCancelCapture}>
              <StopCircle size={18} /> Cancel Stream Capture
            </button>
          ) : (
            <button className="rf-btn rf-btn-primary" style={{ width: '100%' }} onClick={handleStartCapture}>
              <Play size={18} /> Start Stream Capture
            </button>
          )}

          {/* Progress Bar */}
          {(captureStatus.is_running || captureStatus.progress > 0) && (
            <div style={{ marginTop: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
                <span>Capture Progress {captureStatus.is_running ? '(Active)' : '(Finished)'}</span>
                <span>{captureStatus.progress} / {captureStatus.total || numFrames} frames</span>
              </div>
              <div style={{ height: '8px', background: 'var(--bg-input)', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${Math.min(100, (captureStatus.progress / (captureStatus.total || numFrames)) * 100)}%`,
                  background: captureStatus.is_running
                    ? 'linear-gradient(90deg, var(--accent-cyan), var(--accent-blue))'
                    : 'var(--accent-success)',
                  transition: 'width 0.3s'
                }}></div>
              </div>
            </div>
          )}
        </div>

        {/* Auto-Label Assistant Card with Custom Upload & NMS Overlap Threshold Controls */}
        <div className="rf-card" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#FFF', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <Wand2 size={20} color="var(--accent-purple)" /> YOLO AI Auto-Label Assistant
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
            Pre-annotate captured frames from <code>dataset_raw</code> using heavy YOLO models (e.g. <b>YOLOv8x</b>) with Class-Agnostic NMS deduplication.
          </p>

          {/* Model Selector Dropdown & Custom Model Upload */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
              Select YOLO Model Weights (e.g. Heavy Model X for Labeling)
            </label>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
              <select
                className="rf-input"
                style={{ flex: 1 }}
                value={modelPath}
                onChange={(e) => setModelPath(e.target.value)}
              >
                {availableModels.map((m, i) => (
                  <option key={i} value={m}>{m}</option>
                ))}
              </select>

              <label className="rf-btn rf-btn-secondary" style={{ cursor: 'pointer', whiteSpace: 'nowrap' }}>
                <UploadCloud size={16} color="var(--accent-purple)" /> Upload .pt Model
                <input
                  type="file"
                  accept=".pt"
                  style={{ display: 'none' }}
                  onChange={handleModelFileUpload}
                  disabled={uploading}
                />
              </label>
            </div>

            {uploadMsg && (
              <div style={{ fontSize: '0.78rem', color: 'var(--accent-success)', display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '12px' }}>
                <CheckCircle size={14} /> {uploadMsg}
              </div>
            )}

            {/* Optimal Traffic Preset Quick Button */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600 }}>Threshold Controls:</span>
              <button
                className="rf-btn rf-btn-secondary"
                style={{ padding: '4px 8px', fontSize: '0.75rem', color: 'var(--accent-cyan)', borderColor: 'rgba(6, 182, 212, 0.4)' }}
                onClick={applyOptimalPreset}
                title="Set Conf 0.20 & IoU 0.40 for 100% Traffic Detection with Zero Overlap"
              >
                <Zap size={13} /> Optimal Traffic Preset (Conf: 0.20, IoU: 0.40)
              </button>
            </div>

            {/* Threshold Sliders to Eliminate Overlapping Bounding Boxes */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', background: 'var(--bg-input)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)', marginBottom: '16px' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Confidence (Conf):</span>
                  <span style={{ color: 'var(--accent-cyan)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{parseFloat(autolabelConf).toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.80"
                  step="0.05"
                  value={autolabelConf}
                  onChange={(e) => setAutolabelConf(e.target.value)}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
                <span style={{ fontSize: '0.7rem', color: 'var(--text-subtle)' }}>Lower = detect small/distant cars</span>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>IoU Overlap NMS:</span>
                  <span style={{ color: 'var(--accent-purple)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>{parseFloat(autolabelIou).toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.80"
                  step="0.05"
                  value={autolabelIou}
                  onChange={(e) => setAutolabelIou(e.target.value)}
                  style={{ width: '100%', accentColor: 'var(--accent-purple)' }}
                />
                <span style={{ fontSize: '0.7rem', color: 'var(--text-subtle)' }}>Lower = zero duplicate overlap</span>
              </div>
            </div>
          </div>

          {autolabelStatus.is_running ? (
            <button className="rf-btn rf-btn-danger" style={{ width: '100%' }} onClick={handleCancelAutolabel}>
              <StopCircle size={18} /> Cancel Auto-Label
            </button>
          ) : (
            <button className="rf-btn rf-btn-secondary" style={{ width: '100%', borderColor: 'var(--accent-purple)', color: '#FFF' }} onClick={handleStartAutolabel}>
              <Wand2 size={18} color="var(--accent-purple)" /> Start Auto-Labeling with Selected Model
            </button>
          )}

          {(autolabelStatus.is_running || autolabelStatus.progress > 0) && (
            <div style={{ marginTop: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
                <span>Auto-Label Progress {autolabelStatus.is_running ? '(Active)' : '(Finished)'}</span>
                <span>{autolabelStatus.progress} / {autolabelStatus.total || 0} images</span>
              </div>
              <div style={{ height: '8px', background: 'var(--bg-input)', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${Math.min(100, (autolabelStatus.progress / (autolabelStatus.total || 1)) * 100)}%`,
                  background: autolabelStatus.is_running
                    ? 'linear-gradient(90deg, var(--accent-purple), var(--accent-pink))'
                    : 'var(--accent-success)',
                  transition: 'width 0.3s'
                }}></div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right Column: Live Terminal Log Stream */}
      <div className="rf-card" style={{ padding: '20px' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#FFF', marginBottom: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Live Console Output & Background Activity Logs</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--accent-success)', fontFamily: 'var(--font-mono)' }}>● Auto-polling active</span>
        </h3>
        <div style={{
          background: '#06080E',
          padding: '16px',
          borderRadius: '8px',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.8rem',
          color: 'var(--text-muted)',
          maxHeight: '180px',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '4px'
        }}>
          {allLogs.length > 0 ? (
            allLogs.map((log, index) => (
              <div key={index} style={{ color: log.includes('ERROR') ? 'var(--accent-danger)' : log.includes('SUCCESS') ? 'var(--accent-success)' : 'var(--text-muted)' }}>
                {log}
              </div>
            ))
          ) : (
            <div style={{ color: 'var(--text-subtle)' }}>No active task logs yet. Click 'Start Stream Capture' or 'Start Auto-Labeling' to view live activity.</div>
          )}
        </div>
      </div>
    </div>
  );
}

import React, { useState } from 'react';
import { Database, Sliders, CheckCircle2, FileText, ArrowRight, Layers } from 'lucide-react';

export default function DatasetView({ setActiveTab }) {
  const [sourceDir, setSourceDir] = useState('dataset_manual');
  const [splitMode, setSplitMode] = useState('3way'); // '2way' (Train/Val) or '3way' (Train/Val/Test)
  
  const [trainPercent, setTrainPercent] = useState(70);
  const [valPercent, setValPercent] = useState(20);
  const [testPercent, setTestPercent] = useState(10);
  
  const [baseDest, setBaseDest] = useState('dataset');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSplitModeChange = (mode) => {
    setSplitMode(mode);
    if (mode === '2way') {
      setTrainPercent(80);
      setValPercent(20);
      setTestPercent(0);
    } else {
      setTrainPercent(70);
      setValPercent(20);
      setTestPercent(10);
    }
  };

  const handleSplitDataset = async () => {
    setLoading(true);
    setResult(null);

    const total = trainPercent + valPercent + (splitMode === '3way' ? testPercent : 0);
    if (total <= 0) {
      alert('Total ratio must be greater than 0%');
      setLoading(false);
      return;
    }

    const train_ratio = trainPercent / total;
    const val_ratio = valPercent / total;
    const test_ratio = splitMode === '3way' ? (testPercent / total) : 0.0;

    try {
      const res = await fetch('http://localhost:8000/api/dataset/split', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_dir: sourceDir,
          train_ratio: train_ratio,
          val_ratio: val_ratio,
          test_ratio: test_ratio,
          base_dest: baseDest,
          class_names: ["car", "motorcycle", "bus", "truck"]
        })
      });
      const data = await res.json();
      if (!res.ok) {
        alert(data.detail || 'Dataset split failed');
      } else {
        setResult(data);
      }
    } catch (e) {
      alert('Error during dataset split: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '28px', maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div className="rf-card" style={{ padding: '28px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 600, color: '#FFF', display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
          <Database size={22} color="var(--accent-warning)" /> Dataset Preparation & YAML Config Exporter
        </h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginBottom: '24px' }}>
          Split labeled images from <code>{sourceDir}</code> into Train, Validation, and optional Test sets, and generate the official Ultralytics YOLO <code>traffic.yaml</code> file.
        </p>

        {/* Source Folder */}
        <div style={{ marginBottom: '20px' }}>
          <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)', display: 'block', marginBottom: '6px' }}>Source Dataset Folder</label>
          <input
            type="text"
            className="rf-input"
            value={sourceDir}
            onChange={(e) => setSourceDir(e.target.value)}
          />
        </div>

        {/* Split Mode Selector */}
        <div style={{ marginBottom: '20px' }}>
          <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)', display: 'block', marginBottom: '8px' }}>
            Dataset Split Mode
          </label>
          <div style={{ display: 'flex', gap: '12px' }}>
            <button
              className={`rf-btn ${splitMode === '2way' ? 'rf-btn-primary' : 'rf-btn-secondary'}`}
              onClick={() => handleSplitModeChange('2way')}
              style={{ flex: 1 }}
            >
              Train / Val (2-Way Split)
            </button>
            <button
              className={`rf-btn ${splitMode === '3way' ? 'rf-btn-primary' : 'rf-btn-secondary'}`}
              onClick={() => handleSplitModeChange('3way')}
              style={{ flex: 1, borderColor: 'var(--accent-purple)' }}
            >
              Train / Val / Test (3-Way Split)
            </button>
          </div>
        </div>

        {/* Interactive Percentage Controls */}
        <div style={{ marginBottom: '24px', background: 'var(--bg-input)', padding: '20px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
          <div style={{ display: 'grid', gridTemplateColumns: splitMode === '3way' ? '1fr 1fr 1fr' : '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
            {/* Train Input */}
            <div>
              <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--accent-warning)', display: 'block', marginBottom: '4px' }}>
                Train Set (%)
              </label>
              <input
                type="number"
                className="rf-input"
                min="10"
                max="95"
                value={trainPercent}
                onChange={(e) => setTrainPercent(Math.max(0, parseInt(e.target.value) || 0))}
              />
            </div>

            {/* Val Input */}
            <div>
              <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--accent-cyan)', display: 'block', marginBottom: '4px' }}>
                Validation Set (%)
              </label>
              <input
                type="number"
                className="rf-input"
                min="5"
                max="90"
                value={valPercent}
                onChange={(e) => setValPercent(Math.max(0, parseInt(e.target.value) || 0))}
              />
            </div>

            {/* Test Input (Only if 3-Way split enabled) */}
            {splitMode === '3way' && (
              <div>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--accent-pink)', display: 'block', marginBottom: '4px' }}>
                  Test Set (%)
                </label>
                <input
                  type="number"
                  className="rf-input"
                  min="0"
                  max="90"
                  value={testPercent}
                  onChange={(e) => setTestPercent(Math.max(0, parseInt(e.target.value) || 0))}
                />
              </div>
            )}
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', fontWeight: 600 }}>
            <span style={{ color: 'var(--text-muted)' }}>Split Ratio Summary:</span>
            <span style={{ color: '#FFF' }}>
              {trainPercent}% Train • {valPercent}% Val {splitMode === '3way' ? `• ${testPercent}% Test` : ''}
            </span>
          </div>
        </div>

        {/* Output Base Directory */}
        <div style={{ marginBottom: '24px' }}>
          <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)', display: 'block', marginBottom: '6px' }}>Destination Directory</label>
          <input
            type="text"
            className="rf-input"
            value={baseDest}
            onChange={(e) => setBaseDest(e.target.value)}
          />
          <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)', marginTop: '4px', display: 'block' }}>
            Will generate: <code>dataset/train/</code>, <code>dataset/val/</code>{splitMode === '3way' ? ', dataset/test/' : ''}, and <code>dataset/traffic.yaml</code>.
          </span>
        </div>

        <button
          className="rf-btn rf-btn-primary"
          style={{ width: '100%', padding: '12px', background: 'linear-gradient(135deg, #F59E0B, #D97706)', fontSize: '0.95rem' }}
          onClick={handleSplitDataset}
          disabled={loading}
        >
          {loading ? 'Processing Split...' : `Generate ${splitMode === '3way' ? 'Train/Val/Test' : 'Train/Val'} Split & traffic.yaml`}
        </button>
      </div>

      {/* Result Card */}
      {result && (
        <div className="rf-card" style={{ padding: '24px', borderLeft: '4px solid var(--accent-success)' }}>
          <h4 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--accent-success)', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <CheckCircle2 size={20} /> Dataset Split Completed Successfully
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: result.summary.test_count > 0 ? '1fr 1fr 1fr 1fr' : '1fr 1fr 1fr', gap: '16px', marginBottom: '16px', background: 'var(--bg-input)', padding: '16px', borderRadius: '8px' }}>
            <div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-subtle)' }}>Total Images</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#FFF' }}>{result.summary.total_images}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-subtle)' }}>Train Set</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--accent-warning)' }}>{result.summary.train_count}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-subtle)' }}>Val Set</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>{result.summary.val_count}</div>
            </div>
            {result.summary.test_count > 0 && (
              <div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-subtle)' }}>Test Set</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--accent-pink)' }}>{result.summary.test_count}</div>
              </div>
            )}
          </div>

          <div style={{ marginBottom: '16px', fontSize: '0.85rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            <FileText size={16} inline style={{ marginRight: '6px' }} /> YAML File: {result.summary.yaml_path}
          </div>

          <button className="rf-btn rf-btn-primary" onClick={() => setActiveTab('train')}>
            Proceed to YOLO Training Hub <ArrowRight size={16} />
          </button>
        </div>
      )}
    </div>
  );
}

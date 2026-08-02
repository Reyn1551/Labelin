import React, { useState, useEffect, useRef } from 'react';
import { MousePointer, Square, Save, ChevronLeft, ChevronRight, Wand2, Trash2, Undo, Redo, ZoomIn, ZoomOut, RotateCcw, Tag, RefreshCw } from 'lucide-react';

const CLASSES = [
  { id: 0, name: 'car', color: '#3B82F6' },
  { id: 1, name: 'motorcycle', color: '#10B981' },
  { id: 2, name: 'bus', color: '#F59E0B' },
  { id: 3, name: 'truck', color: '#EC4899' }
];

export default function AnnotateView({ activeTab }) {
  const [folder, setFolder] = useState('unlabeled');
  const [imageList, setImageList] = useState([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [selectedClass, setSelectedClass] = useState(0);

  const [boxes, setBoxes] = useState([]);
  const [selectedBoxIdx, setSelectedBoxIdx] = useState(null);
  const [savedStatus, setSavedStatus] = useState(false);

  // Undo / Redo History State
  const [history, setHistory] = useState([]);
  const [historyIdx, setHistoryIdx] = useState(-1);

  // Zoom & Pan State (Capped at 1:1 native image resolution)
  const [zoomLevel, setZoomLevel] = useState(1.0);
  const [maxZoom, setMaxZoom] = useState(2.5);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  // Canvas interaction state
  const [toolMode, setToolMode] = useState('draw'); // 'draw' or 'select'
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState({ x: 0, y: 0 });
  const [currentRect, setCurrentRect] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(new Image());
  const activeFilenameRef = useRef(null);

  // Client-Side Zero-Latency Caching & Background Pre-fetcher
  const labelCache = useRef({});
  const imagePreloadCache = useRef({});

  // Auto-sync images when switching to annotate tab or folder changes
  useEffect(() => {
    fetchImages(folder);
  }, [folder]);

  useEffect(() => {
    if (activeTab === 'annotate') {
      fetchImages(folder);
      const interval = setInterval(() => {
        fetchImages(folder);
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [activeTab, folder]);

  useEffect(() => {
    if (imageList.length > 0 && currentIdx < imageList.length) {
      const targetFilename = imageList[currentIdx];
      // ONLY LOAD IMAGE AND CANVAS IF THE FILENAME HAS CHANGED (PREVENTS 4-SECOND BLINKING)
      if (activeFilenameRef.current !== `${folder}_${targetFilename}`) {
        loadImageAndLabels(targetFilename);
      }
      prefetchAdjacent(currentIdx, folder, imageList);
    }
  }, [currentIdx, imageList, folder]);

  // Non-passive wheel event listener to guarantee smooth Mouse Wheel Zooming
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheelZoom = (e) => {
      e.preventDefault();
      if (e.deltaY < 0) {
        setZoomLevel((prev) => Math.min(maxZoom, parseFloat((prev + 0.15).toFixed(2))));
      } else {
        setZoomLevel((prev) => Math.max(1.0, parseFloat((prev - 0.15).toFixed(2))));
      }
    };

    container.addEventListener('wheel', handleWheelZoom, { passive: false });
    return () => container.removeEventListener('wheel', handleWheelZoom);
  }, [maxZoom]);

  // Fast Cached Label Fetcher
  const fetchLabelsCached = async (targetFolder, filename) => {
    const key = `${targetFolder}_${filename}`;
    if (labelCache.current[key] !== undefined) {
      return labelCache.current[key];
    }
    try {
      const res = await fetch(`http://localhost:8000/api/annotate/labels/${targetFolder}/${filename}`);
      const data = await res.json();
      const fresh = data.boxes || [];
      labelCache.current[key] = fresh;
      return fresh;
    } catch (e) {
      console.error('Error fetching labels:', e);
      return [];
    }
  };

  // Background Pre-loader for Next & Prev Images (Zero Latency Navigation)
  const prefetchAdjacent = (centerIdx, targetFolder, images) => {
    const adjacentIndices = [centerIdx + 1, centerIdx + 2, centerIdx - 1];
    adjacentIndices.forEach(idx => {
      if (idx >= 0 && idx < images.length) {
        const fn = images[idx];
        fetchLabelsCached(targetFolder, fn);
        const imgKey = `${targetFolder}_${fn}`;
        if (!imagePreloadCache.current[imgKey]) {
          const preImg = new Image();
          preImg.crossOrigin = 'anonymous';
          preImg.src = `http://localhost:8000/api/annotate/image_file/${targetFolder}/${fn}`;
          imagePreloadCache.current[imgKey] = preImg;
        }
      }
    });
  };

  // Clamp Panning Offset strictly within container bounds to ensure ZERO blackspace
  const clampPanOffset = (targetX, targetY, zoom) => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return { x: targetX, y: targetY };

    if (zoom <= 1.0) {
      return { x: 0, y: 0 };
    }

    const cRect = container.getBoundingClientRect();
    const canvasBounding = canvas.getBoundingClientRect();

    const zoomedW = canvasBounding.width;
    const zoomedH = canvasBounding.height;

    const extraX = Math.max(0, (zoomedW - cRect.width) / 2);
    const extraY = Math.max(0, (zoomedH - cRect.height) / 2);

    return {
      x: Math.max(-extraX, Math.min(extraX, targetX)),
      y: Math.max(-extraY, Math.min(extraY, targetY))
    };
  };

  useEffect(() => {
    setPanOffset(prev => clampPanOffset(prev.x, prev.y, zoomLevel));
  }, [zoomLevel]);

  // Push state to undo/redo history
  const pushToHistory = (newBoxes) => {
    setBoxes(newBoxes);
    const currentFilename = imageList[currentIdx];
    if (currentFilename) {
      labelCache.current[`${folder}_${currentFilename}`] = newBoxes;
    }
    setHistory(prev => {
      const sliced = prev.slice(0, historyIdx + 1);
      return [...sliced, newBoxes];
    });
    setHistoryIdx(prev => prev + 1);
  };

  const handleUndo = () => {
    if (historyIdx > 0) {
      const prevIdx = historyIdx - 1;
      setHistoryIdx(prevIdx);
      const targetBoxes = history[prevIdx];
      setBoxes(targetBoxes);
      const currentFilename = imageList[currentIdx];
      if (currentFilename) {
        labelCache.current[`${folder}_${currentFilename}`] = targetBoxes;
      }
      setSelectedBoxIdx(null);
    }
  };

  const handleRedo = () => {
    if (historyIdx < history.length - 1) {
      const nextIdx = historyIdx + 1;
      setHistoryIdx(nextIdx);
      const targetBoxes = history[nextIdx];
      setBoxes(targetBoxes);
      const currentFilename = imageList[currentIdx];
      if (currentFilename) {
        labelCache.current[`${folder}_${currentFilename}`] = targetBoxes;
      }
      setSelectedBoxIdx(null);
    }
  };

  // Global hotkeys
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
        e.preventDefault();
        if (e.shiftKey) {
          handleRedo();
        } else {
          handleUndo();
        }
      } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Y')) {
        e.preventDefault();
        handleRedo();
      } else if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') {
        prevImage();
      } else if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') {
        nextImage();
      } else if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        saveAnnotations();
      } else if (e.key === 'Delete' || e.key === 'Backspace' || e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        if (selectedBoxIdx !== null) {
          deleteSelectedBox();
        }
      } else if (e.key >= '1' && e.key <= '4') {
        const clsIndex = parseInt(e.key) - 1;
        if (clsIndex < CLASSES.length) {
          setSelectedClass(clsIndex);
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentIdx, imageList, boxes, selectedBoxIdx, history, historyIdx]);

  const fetchImages = async (targetFolder) => {
    try {
      const res = await fetch(`http://localhost:8000/api/annotate/images?folder=${targetFolder}`);
      const data = await res.json();
      const newImages = data.images || [];
      
      // ONLY UPDATE STATE IF IMAGE LIST LIST OF FILENAMES ACTUALLY CHANGED
      setImageList(prevList => {
        if (prevList.length === newImages.length && prevList.every((val, index) => val === newImages[index])) {
          return prevList; // Return same reference to prevent re-renders / blinking
        }
        return newImages;
      });

      if (currentIdx >= newImages.length && newImages.length > 0) {
        setCurrentIdx(newImages.length - 1);
      }
    } catch (e) {
      console.error('Failed to fetch images:', e);
    }
  };

  const loadImageAndLabels = async (filename) => {
    activeFilenameRef.current = `${folder}_${filename}`;
    setSavedStatus(false);
    setSelectedBoxIdx(null);
    setZoomLevel(1.0);
    setPanOffset({ x: 0, y: 0 });

    // IMMEDIATELY CLEAR CANVAS AND BOXES TO PREVENT OLD IMAGE / OLD BOX LEAKS
    setBoxes([]);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    const freshBoxes = await fetchLabelsCached(folder, filename);
    const imgUrl = `http://localhost:8000/api/annotate/image_file/${folder}/${filename}`;

    const newImg = new Image();
    newImg.crossOrigin = 'anonymous';
    newImg.src = imgUrl;

    newImg.onload = () => {
      imgRef.current = newImg;

      if (canvas && newImg.width && newImg.height) {
        canvas.width = newImg.width;
        canvas.height = newImg.height;

        if (containerRef.current) {
          const cRect = containerRef.current.getBoundingClientRect();
          const scaleW = cRect.width / newImg.width;
          const scaleH = cRect.height / newImg.height;
          const fitScale = Math.min(scaleW, scaleH);
          if (fitScale > 0) {
            const nativeZoomCap = parseFloat((1 / fitScale).toFixed(2));
            setMaxZoom(Math.max(1.0, nativeZoomCap));
          }
        }
      }

      setBoxes(freshBoxes);
      setHistory([freshBoxes]);
      setHistoryIdx(0);
      drawCanvasWithBoxes(freshBoxes, newImg);
    };
  };

  const drawCanvasWithBoxes = (targetBoxes = boxes, targetImg = imgRef.current) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const img = targetImg || imgRef.current;

    if (!img.width || !img.height) return;

    if (canvas.width !== img.width || canvas.height !== img.height) {
      canvas.width = img.width;
      canvas.height = img.height;
    }

    // Draw background image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Draw target boxes
    targetBoxes.forEach((box, index) => {
      const isSelected = index === selectedBoxIdx;
      const cls = CLASSES.find(c => c.id === box.cls_id) || CLASSES[0];

      const bw = box.width * canvas.width;
      const bh = box.height * canvas.height;
      const bx = (box.x_center * canvas.width) - (bw / 2);
      const by = (box.y_center * canvas.height) - (bh / 2);

      ctx.lineWidth = isSelected ? 3.5 : 2;
      ctx.strokeStyle = isSelected ? '#FFFFFF' : cls.color;
      ctx.fillStyle = isSelected ? `${cls.color}35` : `${cls.color}20`;

      ctx.fillRect(bx, by, bw, bh);
      ctx.strokeRect(bx, by, bw, bh);

      // Draw label header badge
      ctx.fillStyle = cls.color;
      ctx.font = 'bold 12px Inter, sans-serif';
      const text = isSelected ? `★ ${cls.name}` : `${cls.name}`;
      const textWidth = ctx.measureText(text).width;
      const badgeY = by > 22 ? by - 22 : by;

      ctx.fillRect(bx, badgeY, textWidth + 10, 18);

      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(text, bx + 5, badgeY + 13);
    });

    // Draw Active Drawing Preview Rectangle while dragging click
    if (currentRect && (currentRect.w > 2 || currentRect.h > 2)) {
      const cls = CLASSES[selectedClass];

      ctx.lineWidth = 3;
      ctx.strokeStyle = '#000000a0';
      ctx.strokeRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h);

      ctx.lineWidth = 2;
      ctx.strokeStyle = cls.color;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h);
      ctx.setLineDash([]);
    }

    // Draw Crosshair guides
    if (mousePos.x > 0 && mousePos.y > 0) {
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = '#000000';
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(mousePos.x, 0);
      ctx.lineTo(mousePos.x, canvas.height);
      ctx.moveTo(0, mousePos.y);
      ctx.lineTo(canvas.width, mousePos.y);
      ctx.stroke();
    }
  };

  const drawCanvas = () => {
    drawCanvasWithBoxes(boxes, imgRef.current);
  };

  useEffect(() => {
    drawCanvas();
  }, [boxes, selectedBoxIdx, currentRect, mousePos]);

  const getCanvasCoords = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  };

  const handleMouseDown = (e) => {
    if (e.button === 2) {
      e.preventDefault();
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
      return;
    }

    if (e.button !== 0) return;

    const coords = getCanvasCoords(e);
    const canvas = canvasRef.current;

    let clickedBoxIdx = null;
    boxes.forEach((box, idx) => {
      const bw = box.width * canvas.width;
      const bh = box.height * canvas.height;
      const bx = (box.x_center * canvas.width) - (bw / 2);
      const by = (box.y_center * canvas.height) - (bh / 2);

      if (coords.x >= bx && coords.x <= bx + bw && coords.y >= by && coords.y <= by + bh) {
        clickedBoxIdx = idx;
      }
    });

    if (clickedBoxIdx !== null) {
      setSelectedBoxIdx(clickedBoxIdx);
    } else if (toolMode === 'select') {
      setSelectedBoxIdx(null);
    }

    if (toolMode === 'draw') {
      setIsDrawing(true);
      setDrawStart(coords);
      setCurrentRect({ x: coords.x, y: coords.y, w: 0, h: 0 });
    }
  };

  const handleMouseMove = (e) => {
    if (isPanning) {
      const dx = e.clientX - panStart.x;
      const dy = e.clientY - panStart.y;
      setPanOffset(prev => clampPanOffset(prev.x + dx, prev.y + dy, zoomLevel));
      setPanStart({ x: e.clientX, y: e.clientY });
      return;
    }

    const coords = getCanvasCoords(e);
    setMousePos(coords);

    if (isDrawing && toolMode === 'draw') {
      const w = coords.x - drawStart.x;
      const h = coords.y - drawStart.y;
      setCurrentRect({
        x: w < 0 ? coords.x : drawStart.x,
        y: h < 0 ? coords.y : drawStart.y,
        w: Math.abs(w),
        h: Math.abs(h)
      });
    }
  };

  const handleMouseUp = (e) => {
    if (isPanning || (e && e.button === 2)) {
      setIsPanning(false);
      return;
    }

    if (isDrawing && currentRect && currentRect.w > 12 && currentRect.h > 12) {
      const canvas = canvasRef.current;
      const xc = (currentRect.x + currentRect.w / 2) / canvas.width;
      const yc = (currentRect.y + currentRect.h / 2) / canvas.height;
      const bw = currentRect.w / canvas.width;
      const bh = currentRect.h / canvas.height;

      const newBox = {
        cls_id: selectedClass,
        x_center: xc,
        y_center: yc,
        width: bw,
        height: bh
      };

      const nextBoxes = [...boxes, newBox];
      pushToHistory(nextBoxes);
      setSelectedBoxIdx(nextBoxes.length - 1);
    }
    setIsDrawing(false);
    setCurrentRect(null);
  };

  const resetZoomAndPan = () => {
    setZoomLevel(1.0);
    setPanOffset({ x: 0, y: 0 });
  };

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e) => {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 0.15 : -0.15;
      
      setZoomLevel(prevZoom => {
        const newZoom = Math.max(1.0, Math.min(maxZoom, parseFloat((prevZoom + delta).toFixed(2))));
        if (newZoom === prevZoom) return prevZoom;

        const cRect = container.getBoundingClientRect();
        const mouseX = e.clientX - (cRect.left + cRect.width / 2);
        const mouseY = e.clientY - (cRect.top + cRect.height / 2);

        const zoomFactor = newZoom / prevZoom;

        setPanOffset(prevPan => {
          const newPanX = mouseX - zoomFactor * (mouseX - prevPan.x);
          const newPanY = mouseY - zoomFactor * (mouseY - prevPan.y);
          return clampPanOffset(newPanX, newPanY, newZoom);
        });

        return newZoom;
      });
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, [maxZoom]);

  const deleteSelectedBox = () => {
    if (selectedBoxIdx !== null) {
      const nextBoxes = boxes.filter((_, idx) => idx !== selectedBoxIdx);
      pushToHistory(nextBoxes);
      setSelectedBoxIdx(null);
    }
  };

  const saveAnnotationsForIdx = async (idxToSave) => {
    if (imageList.length === 0 || idxToSave < 0 || idxToSave >= imageList.length) return;
    const imgFilename = imageList[idxToSave];
    try {
      await fetch('http://localhost:8000/api/annotate/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_filename: imgFilename,
          boxes: boxes
        })
      });
      labelCache.current[`${folder}_${imgFilename}`] = boxes;
      setSavedStatus(true);
    } catch (e) {
      console.error('Auto-save failed:', e);
    }
  };

  const saveAnnotations = async () => {
    await saveAnnotationsForIdx(currentIdx);
  };

  const nextImage = async () => {
    if (currentIdx < imageList.length - 1) {
      await saveAnnotationsForIdx(currentIdx);
      setCurrentIdx(currentIdx + 1);
    }
  };

  const prevImage = async () => {
    if (currentIdx > 0) {
      await saveAnnotationsForIdx(currentIdx);
      setCurrentIdx(currentIdx - 1);
    }
  };

  const currentFilename = imageList[currentIdx] || 'No Image Loaded';

  return (
    <div style={{ padding: '16px', height: 'calc(100vh - 64px)', display: 'flex', flexDirection: 'column', gap: '12px', overflow: 'hidden' }}>
      {/* Top Main Toolbar - Responsive & Compact */}
      <div className="rf-card" style={{ padding: '10px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', flexShrink: 0, flexWrap: 'wrap' }}>
        
        {/* Group 1: Folder, Sync, Image Navigation & Prominent Zoom Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexShrink: 0 }}>
          <select
            className="rf-input"
            style={{ width: '160px', flexShrink: 0 }}
            value={folder}
            onChange={(e) => {
              setFolder(e.target.value);
              setCurrentIdx(0);
            }}
          >
            <option value="unlabeled">Unlabeled Images</option>
            <option value="labeled">Labeled Images</option>
          </select>

          <button
            className="rf-btn rf-btn-secondary"
            style={{ padding: '6px 10px', fontSize: '0.8rem', color: 'var(--accent-cyan)', borderColor: 'rgba(6, 182, 212, 0.4)', flexShrink: 0 }}
            onClick={() => {
              labelCache.current = {};
              fetchImages(folder);
            }}
            title="Sync Latest Captured Images"
          >
            <RefreshCw size={14} /> Sync
          </button>

          {/* Navigation Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
            <button className="rf-btn rf-btn-secondary" style={{ padding: '6px 10px', flexShrink: 0 }} onClick={prevImage} disabled={currentIdx === 0}>
              <ChevronLeft size={16} /> Prev (A)
            </button>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap', flexShrink: 0, padding: '0 2px' }}>
              {imageList.length > 0 ? `${currentIdx + 1}/${imageList.length}` : '0/0'}
            </span>
            <button className="rf-btn rf-btn-secondary" style={{ padding: '6px 10px', flexShrink: 0 }} onClick={nextImage} disabled={currentIdx >= imageList.length - 1}>
              Next (D) <ChevronRight size={16} />
            </button>
          </div>

          {/* Prominent Zoom Controls (Always Visible Right Next to Navigation) */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', background: 'var(--bg-input)', padding: '2px 6px', borderRadius: '8px', border: '1px solid var(--border-color)', flexShrink: 0 }}>
            <button
              className="rf-btn rf-btn-secondary"
              style={{ padding: '4px 6px', border: 'none' }}
              onClick={() => setZoomLevel(prev => Math.max(1.0, parseFloat((prev - 0.2).toFixed(2))))}
              title="Zoom Out"
              disabled={zoomLevel <= 1.0}
            >
              <ZoomOut size={15} />
            </button>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--accent-cyan)', minWidth: '46px', textAlign: 'center', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>
              {Math.round(zoomLevel * 100)}%
            </span>
            <button
              className="rf-btn rf-btn-secondary"
              style={{ padding: '4px 6px', border: 'none' }}
              onClick={() => setZoomLevel(prev => Math.min(maxZoom, parseFloat((prev + 0.2).toFixed(2))))}
              title={`Zoom In (Max Native Res: ${Math.round(maxZoom * 100)}%)`}
              disabled={zoomLevel >= maxZoom}
            >
              <ZoomIn size={15} />
            </button>
            <button
              className="rf-btn rf-btn-secondary"
              style={{ padding: '4px 6px', border: 'none', color: 'var(--text-subtle)' }}
              onClick={resetZoomAndPan}
              title="Reset Zoom & Pan (Fit to Screen)"
            >
              <RotateCcw size={14} />
            </button>
          </div>
        </div>

        {/* Group 2: Center Class Selector Badges */}
        <div style={{ display: 'flex', gap: '6px', flexShrink: 0 }}>
          {CLASSES.map((cls, idx) => (
            <button
              key={cls.id}
              onClick={() => setSelectedClass(cls.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '5px 10px',
                borderRadius: '8px',
                border: selectedClass === cls.id ? `2px solid ${cls.color}` : '1px solid var(--border-color)',
                background: selectedClass === cls.id ? `${cls.color}25` : 'var(--bg-input)',
                color: '#FFF',
                fontSize: '0.78rem',
                fontWeight: 600,
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                flexShrink: 0
              }}
            >
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: cls.color }}></span>
              {idx + 1}. {cls.name}
            </button>
          ))}
        </div>

        {/* Group 3: Right Tools, Undo/Redo & Save Button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap', marginRight: '4px' }}>
            {currentFilename} {savedStatus && <span style={{ color: 'var(--accent-success)' }}>[SAVED]</span>}
          </span>

          <button
            className={`rf-btn ${toolMode === 'draw' ? 'rf-btn-primary' : 'rf-btn-secondary'}`}
            style={{ padding: '6px 10px', flexShrink: 0 }}
            onClick={() => setToolMode('draw')}
            title="Draw Box (W)"
          >
            <Square size={15} /> Draw
          </button>
          <button
            className={`rf-btn ${toolMode === 'select' ? 'rf-btn-primary' : 'rf-btn-secondary'}`}
            style={{ padding: '6px 10px', flexShrink: 0 }}
            onClick={() => setToolMode('select')}
            title="Select Box"
          >
            <MousePointer size={15} /> Select
          </button>

          <div style={{ display: 'flex', gap: '4px', flexShrink: 0 }}>
            <button
              className="rf-btn rf-btn-secondary"
              style={{ padding: '6px 8px' }}
              onClick={handleUndo}
              disabled={historyIdx <= 0}
              title="Undo (Ctrl+Z)"
            >
              <Undo size={15} />
            </button>
            <button
              className="rf-btn rf-btn-secondary"
              style={{ padding: '6px 8px' }}
              onClick={handleRedo}
              disabled={historyIdx >= history.length - 1}
              title="Redo (Ctrl+Y / Ctrl+Shift+Z)"
            >
              <Redo size={15} />
            </button>
          </div>

          <button className="rf-btn rf-btn-primary" style={{ padding: '6px 14px', background: 'linear-gradient(135deg, #10B981, #059669)', flexShrink: 0, whiteSpace: 'nowrap' }} onClick={saveAnnotations}>
            <Save size={15} /> Save (S)
          </button>
        </div>
      </div>

      {/* Main Workspace Area */}
      <div style={{ flex: 1, minHeight: 0, height: '100%', display: 'grid', gridTemplateColumns: '1fr 280px', gap: '16px', overflow: 'hidden' }}>
        {/* Canvas Display Container */}
        <div
          ref={containerRef}
          className="rf-card"
          onContextMenu={(e) => e.preventDefault()}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => setIsPanning(false)}
          style={{
            position: 'relative',
            height: '100%',
            minHeight: 0,
            overflow: 'hidden',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#06080E',
            userSelect: 'none'
          }}
        >
          {imageList.length > 0 ? (
            <div style={{
              transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomLevel})`,
              transformOrigin: 'center center',
              transition: isPanning ? 'none' : 'transform 0.05s ease-out',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <canvas
                ref={canvasRef}
                onContextMenu={(e) => e.preventDefault()}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                style={{
                  cursor: isPanning ? 'grabbing' : toolMode === 'draw' ? 'crosshair' : 'default',
                  maxWidth: '100%',
                  maxHeight: '100%',
                  objectFit: 'contain'
                }}
              />
            </div>
          ) : (
            <div style={{ color: 'var(--text-subtle)', textAlign: 'center' }}>
              <Tag size={48} style={{ opacity: 0.3, marginBottom: '12px' }} />
              <p>No images found in '{folder}'.</p>
              <p style={{ fontSize: '0.8rem', marginTop: '4px' }}>Capture images in Step 1 or switch folder above.</p>
            </div>
          )}
        </div>

        {/* Right Drawer: Bounding Boxes List */}
        <div className="rf-card" style={{ padding: '16px', display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, overflow: 'hidden' }}>
          <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#FFF', marginBottom: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0 }}>
            <span>Bounding Boxes ({boxes.length})</span>
            {selectedBoxIdx !== null && (
              <button style={{ background: 'transparent', border: 'none', color: 'var(--accent-danger)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.75rem' }} onClick={deleteSelectedBox}>
                <Trash2 size={14} /> Delete (Space)
              </button>
            )}
          </h4>

          {/* Scrollable List Container */}
          <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px', paddingRight: '4px' }}>
            {boxes.map((box, idx) => {
              const cls = CLASSES.find(c => c.id === box.cls_id) || CLASSES[0];
              const isSel = idx === selectedBoxIdx;
              return (
                <div
                  key={idx}
                  onClick={() => setSelectedBoxIdx(idx)}
                  style={{
                    padding: '8px 10px',
                    borderRadius: '8px',
                    border: isSel ? `2px solid ${cls.color}` : '1px solid var(--border-color)',
                    background: isSel ? `${cls.color}25` : 'var(--bg-input)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    fontSize: '0.8rem',
                    flexShrink: 0
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: cls.color }}></span>
                    <span style={{ fontWeight: 600, color: '#FFF' }}>{cls.name}</span>
                  </div>
                  <span style={{ color: 'var(--text-subtle)', fontFamily: 'var(--font-mono)', fontSize: '0.72rem' }}>
                    xc:{box.x_center.toFixed(2)} yc:{box.y_center.toFixed(2)}
                  </span>
                </div>
              );
            })}
            {boxes.length === 0 && (
              <div style={{ color: 'var(--text-subtle)', fontSize: '0.8rem', textAlign: 'center', marginTop: '20px' }}>
                No bounding boxes drawn yet.<br />Click and drag on the canvas to add boxes.
              </div>
            )}
          </div>

          <div style={{ marginTop: '12px', borderTop: '1px solid var(--border-color)', paddingTop: '10px', fontSize: '0.73rem', color: 'var(--text-subtle)', flexShrink: 0 }}>
            <div>Controls & Shortcuts:</div>
            <div><code>Right Click Drag</code>: Pan Canvas</div>
            <div><code>Scroll Wheel</code>: Zoom ({Math.round(zoomLevel * 100)}%)</div>
            <div><code>Ctrl+Z / Y</code>: Undo / Redo</div>
            <div><code>S</code>: Save • <code>A / D</code>: Prev / Next</div>
            <div><code>Space / Del</code>: Delete Box • <code>1-4</code>: Class</div>
          </div>
        </div>
      </div>
    </div>
  );
}

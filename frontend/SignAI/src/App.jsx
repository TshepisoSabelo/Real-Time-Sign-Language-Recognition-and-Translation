import { useState } from 'react';
import { Camera, CircleStop, Play, RotateCcw, ScanLine, Hand } from 'lucide-react';

const statusLabels = { ready: 'Ready', active: 'Camera Active', stopped: 'Stopped' };

function StatusIndicator({ status }) {
  return <div className={`status status--${status}`} role="status" aria-live="polite"><span className="status__dot" aria-hidden="true" />{statusLabels[status]}</div>;
}

function Header({ status }) {
  return <header className="header"><div className="brand"><div className="brand__mark"><Hand size={23}/></div><div><h1>SignAi</h1><p>Real-time sign language recognition and translation</p></div></div><StatusIndicator status={status}/></header>;
}

function CameraPreview({ active }) {
  return <section className="card camera-card" aria-labelledby="camera-title"><div className="section-heading"><div><span className="eyebrow">Live input</span><h2 id="camera-title">Camera Preview</h2></div><span className={`live-badge ${active ? 'live-badge--active' : ''}`}><span />{active ? 'Recognition active' : 'Recognition idle'}</span></div><div className={`camera-stage ${active ? 'camera-stage--active' : ''}`}><div className="corner corner--tl"/><div className="corner corner--tr"/><div className="corner corner--bl"/><div className="corner corner--br"/><div className="camera-placeholder"><div className="camera-icon"><Camera size={34}/></div><strong>{active ? 'Camera ready for integration' : 'Camera Preview'}</strong><p>{active ? 'Frontend recognition state is active' : 'Camera feed will appear here'}</p></div>{active && <ScanLine className="scan-icon" size={22} aria-hidden="true"/>}</div></section>;
}

function DetectedSign({ letter }) {
  return <section className="card detected-card" aria-labelledby="detected-title"><div className="section-heading"><div><span className="eyebrow">Latest result</span><h2 id="detected-title">Detected Sign</h2></div></div><div className="detected-letter" aria-live="polite">{letter || '—'}</div><p className="muted">The latest recognized letter will appear here.</p></section>;
}

function TranslationPanel({ translation }) {
  return <section className="card translation-card" aria-labelledby="translation-title"><div className="section-heading"><div><span className="eyebrow">Recognition output</span><h2 id="translation-title">Translation</h2></div><span className="character-count">{translation.length} characters</span></div><div className={`translation-output ${!translation ? 'translation-output--empty' : ''}`} aria-live="polite">{translation || 'Recognized letters will appear here...'}</div></section>;
}

function RecognitionControls({ status, onStart, onStop, onClear }) {
  const active = status === 'active';
  return <section className="controls" aria-label="Recognition controls"><button className="button button--primary" onClick={onStart} disabled={active}><Play size={18} fill="currentColor"/>Start Recognition</button><button className="button button--secondary" onClick={onStop} disabled={!active}><CircleStop size={18}/>Stop Recognition</button><button className="button button--ghost" onClick={onClear}><RotateCcw size={18}/>Clear Translation</button></section>;
}

export default function App() {
  const [status, setStatus] = useState('ready');
  const [detectedLetter, setDetectedLetter] = useState('');
  const [translation, setTranslation] = useState('');
  const clear = () => { setDetectedLetter(''); setTranslation(''); };

  return <div className="app-shell"><Header status={status}/><main><div className="intro"><div><span className="eyebrow">Computer vision interface</span><h2>Translate hand signs into text.</h2><p>Position a hand sign inside the camera frame. Recognition results from your model will be displayed here once the backend is connected.</p></div></div><div className="workspace"><CameraPreview active={status === 'active'}/><DetectedSign letter={detectedLetter}/></div><TranslationPanel translation={translation}/><RecognitionControls status={status} onStart={() => setStatus('active')} onStop={() => setStatus('stopped')} onClear={clear}/><p className="integration-note">Frontend prototype • Camera and ML inference are intentionally not connected yet.</p></main></div>;
}

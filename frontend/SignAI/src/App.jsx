import { useEffect, useRef, useState } from 'react';
import {
  Camera,
  CircleStop,
  Hand,
  Play,
  RotateCcw,
  ScanLine,
} from 'lucide-react';

const STATUS_LABELS = {
  ready: 'Ready',
  connecting: 'Connecting',
  active: 'Connected',
  stopped: 'Stopped',
};

const RECONNECT_DELAY_MS = 1500;
const NO_SIGN = 'None';

const websocketProtocol =
  window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const websocketUrl =
  import.meta.env.VITE_WS_URL ||
  `${websocketProtocol}//${window.location.hostname}:8000/ws`;

function StatusIndicator({ status }) {
  return (
    <div
      className={`status status--${status}`}
      role="status"
      aria-live="polite"
    >
      <span className="status__dot" aria-hidden="true" />
      {STATUS_LABELS[status]}
    </div>
  );
}

function Header({ status }) {
  return (
    <header className="header">
      <div className="brand">
        <div className="brand__mark">
          <Hand size={23} />
        </div>
        <div>
          <h1>SignAi</h1>
          <p>Real-time sign language recognition and translation</p>
        </div>
      </div>
      <StatusIndicator status={status} />
    </header>
  );
}

function CameraPreview({ active }) {
  return (
    <section className="card camera-card" aria-labelledby="camera-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Live input</span>
          <h2 id="camera-title">Camera Preview</h2>
        </div>
        <span className={`live-badge ${active ? 'live-badge--active' : ''}`}>
          <span />
          {active ? 'Recognition active' : 'Recognition idle'}
        </span>
      </div>
      <div className={`camera-stage ${active ? 'camera-stage--active' : ''}`}>
        <div className="corner corner--tl" />
        <div className="corner corner--tr" />
        <div className="corner corner--bl" />
        <div className="corner corner--br" />
        <div className="camera-placeholder">
          <div className="camera-icon">
            <Camera size={34} />
          </div>
          <strong>{active ? 'Prediction stream connected' : 'Camera Preview'}</strong>
          <p>
            {active
              ? 'Live predictions are arriving from the backend'
              : 'Camera feed will appear here'}
          </p>
        </div>
        {active && <ScanLine className="scan-icon" size={22} aria-hidden="true" />}
      </div>
    </section>
  );
}

function DetectedSign({ letter }) {
  return (
    <section className="card detected-card" aria-labelledby="detected-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Latest result</span>
          <h2 id="detected-title">Detected Sign</h2>
        </div>
      </div>
      <div className="detected-letter" aria-live="polite">
        {letter || '—'}
      </div>
      <p className="muted">The latest recognized letter will appear here.</p>
    </section>
  );
}

function TranslationPanel({ translation }) {
  return (
    <section
      className="card translation-card"
      aria-labelledby="translation-title"
    >
      <div className="section-heading">
        <div>
          <span className="eyebrow">Recognition output</span>
          <h2 id="translation-title">Translation</h2>
        </div>
        <span className="character-count">{translation.length} characters</span>
      </div>
      <div
        className={`translation-output ${
          !translation ? 'translation-output--empty' : ''
        }`}
        aria-live="polite"
      >
        {translation || 'Recognized letters will appear here...'}
      </div>
    </section>
  );
}

function RecognitionControls({
  status,
  onStart,
  onStop,
  onClear,
}) {
  const isActive = status === 'active';
  const isConnecting = status === 'connecting';

  return (
    <section className="controls" aria-label="Recognition controls">
      <button
        className="button button--primary"
        onClick={onStart}
        disabled={isActive || isConnecting}
      >
        <Play size={18} fill="currentColor" />
        Start Recognition
      </button>
      <button
        className="button button--secondary"
        onClick={onStop}
        disabled={!isActive && !isConnecting}
      >
        <CircleStop size={18} />
        Stop Recognition
      </button>
      <button className="button button--ghost" onClick={onClear}>
        <RotateCcw size={18} />
        Clear Translation
      </button>
    </section>
  );
}

function isPredictionMessage(value) {
  return (
    value !== null &&
    typeof value === 'object' &&
    typeof value.letter === 'string' &&
    value.letter.length > 0 &&
    typeof value.confidence === 'number' &&
    Number.isFinite(value.confidence) &&
    value.confidence >= 0 &&
    value.confidence <= 1
  );
}

export default function App() {
  const [status, setStatus] = useState('ready');
  const [connectionMessage, setConnectionMessage] = useState(
    'Start recognition after the API and camera processes are running.',
  );
  const [detectedLetter, setDetectedLetter] = useState('');
  const [translation, setTranslation] = useState('');

  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const shouldReconnectRef = useRef(false);
  const lastAppendedLetterRef = useRef(null);

  function handlePrediction(message) {
    let prediction;

    try {
      prediction = JSON.parse(message);
    } catch (error) {
      console.error('The API sent invalid JSON over the WebSocket.', error);
      setConnectionMessage('The API sent invalid prediction data.');
      return;
    }

    if (!isPredictionMessage(prediction)) {
      console.error('The API sent an invalid prediction payload.', prediction);
      setConnectionMessage('The API sent invalid prediction data.');
      return;
    }

    if (prediction.letter === NO_SIGN) {
      setDetectedLetter('');
      lastAppendedLetterRef.current = null;
      return;
    }

    setDetectedLetter(prediction.letter);

    // Only add a letter when it changes
    if (prediction.letter !== lastAppendedLetterRef.current) {
      setTranslation((currentText) => currentText + prediction.letter);
      lastAppendedLetterRef.current = prediction.letter;
    }
  }

  function connectToWebSocket() {
    if (!shouldReconnectRef.current) {
      return;
    }

    setStatus('connecting');
    setConnectionMessage('Connecting to the live prediction stream...');

    let socket;

    try {
      socket = new WebSocket(websocketUrl);
    } catch (error) {
      console.error('Could not create the prediction WebSocket.', error);
      setConnectionMessage('Could not connect. Retrying...');
      reconnectTimerRef.current = window.setTimeout(
        connectToWebSocket,
        RECONNECT_DELAY_MS,
      );
      return;
    }

    socketRef.current = socket;

    socket.onopen = () => {
      if (socketRef.current !== socket) {
        return;
      }

      setStatus('active');
      setConnectionMessage('Receiving live predictions from the camera.');
    };

    socket.onmessage = (event) => {
      if (socketRef.current === socket) {
        handlePrediction(event.data);
      }
    };

    socket.onerror = () => {
      socket.close();
    };

    socket.onclose = () => {
      if (socketRef.current !== socket) {
        return;
      }

      socketRef.current = null;

      if (shouldReconnectRef.current) {
        setStatus('connecting');
        setConnectionMessage('Connection lost. Reconnecting...');
        reconnectTimerRef.current = window.setTimeout(
          connectToWebSocket,
          RECONNECT_DELAY_MS,
        );
      } else {
        setStatus('stopped');
        setConnectionMessage('The prediction stream is stopped.');
      }
    };
  }

  function startRecognition() {
    if (socketRef.current) {
      return;
    }

    shouldReconnectRef.current = true;
    connectToWebSocket();
  }

  function stopRecognition() {
    shouldReconnectRef.current = false;
    window.clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = null;

    const socket = socketRef.current;
    socketRef.current = null;
    if (socket) {
      socket.close();
    }

    setStatus('stopped');
    setConnectionMessage('The prediction stream is stopped.');
  }

  function clearTranslation() {
    setDetectedLetter('');
    setTranslation('');
  }

  useEffect(() => {
    return () => {
      shouldReconnectRef.current = false;
      window.clearTimeout(reconnectTimerRef.current);
      socketRef.current?.close();
    };
  }, []);

  return (
    <div className="app-shell">
      <Header status={status} />

      <main>
        <div className="intro">
          <div>
            <span className="eyebrow">Computer vision interface</span>
            <h2>Translate hand signs into text.</h2>
            <p>
              Position a hand sign inside the camera frame. Live recognition
              results from the backend will appear here.
            </p>
          </div>
        </div>

        <div className="workspace">
          <CameraPreview active={status === 'active'} />
          <DetectedSign letter={detectedLetter} />
        </div>

        <TranslationPanel translation={translation} />

        <RecognitionControls
          status={status}
          onStart={startRecognition}
          onStop={stopRecognition}
          onClear={clearTranslation}
        />

        <p className="integration-note" role="status" aria-live="polite">
          {connectionMessage}
        </p>
      </main>
    </div>
  );
}

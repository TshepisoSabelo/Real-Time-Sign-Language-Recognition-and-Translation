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
  active: 'Live',
  stopped: 'Paused',
};

const RECONNECT_DELAY_MS = 1500;
const NO_SIGN = 'None';
const MAX_FRAME_RATE = 60;
const FRAME_RATE = Math.min(
  MAX_FRAME_RATE,
  Math.max(1, Number(import.meta.env.VITE_FRAME_RATE) || MAX_FRAME_RATE),
);
const MAX_BUFFERED_BYTES = 256 * 1024;

const websocketProtocol =
  window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const websocketUrl =
  import.meta.env.VITE_WS_URL ||
  `${websocketProtocol}//${window.location.hostname}:8000/ws/recognition`;

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
          <p>Helping turn signs into words</p>
        </div>
      </div>
      <StatusIndicator status={status} />
    </header>
  );
}

function CameraPreview({ active, cameraReady, videoRef }) {
  return (
    <section className="card camera-card" aria-labelledby="camera-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Live input</span>
          <h2 id="camera-title">Camera Preview</h2>
        </div>
        <span className={`live-badge ${active ? 'live-badge--active' : ''}`}>
          <span />
          {active ? 'Watching your hands' : 'Waiting for camera'}
        </span>
      </div>
      <div className={`camera-stage ${active ? 'camera-stage--active' : ''}`}>
        <div className="corner corner--tl" />
        <div className="corner corner--tr" />
        <div className="corner corner--bl" />
        <div className="corner corner--br" />
        <video
          ref={videoRef}
          className={`camera-video ${cameraReady ? 'camera-video--visible' : ''}`}
          autoPlay
          playsInline
          muted
          style={{ transform: 'scaleX(-1)' }}
        />
        {!cameraReady && (
          <div className="camera-placeholder">
            <div className="camera-icon">
              <Camera size={34} />
            </div>
            <strong>{active ? 'Your live stream is on' : 'Camera preview'}</strong>
            <p>
              {active
                ? 'Your sign is being read in real time'
                : 'Your camera feed will appear here'}
            </p>
          </div>
        )}
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
          <h2 id="detected-title">Recognized sign</h2>
        </div>
      </div>
      <div className="detected-letter" aria-live="polite">
        {letter || '—'}
      </div>
      <p className="muted">The latest sign the app has picked up will appear here.</p>
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
          <h2 id="translation-title">Your text</h2>
        </div>
        <span className="character-count">{translation.length} characters</span>
      </div>
      <div
        className={`translation-output ${
          !translation ? 'translation-output--empty' : ''
        }`}
        aria-live="polite"
      >
        {translation || 'Your translated words will appear here...'}
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
        Start live recognition
      </button>
      <button
        className="button button--secondary"
        onClick={onStop}
        disabled={!isActive && !isConnecting}
      >
        <CircleStop size={18} />
        Pause
      </button>
      <button className="button button--ghost" onClick={onClear}>
        <RotateCcw size={18} />
        Clear text
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
    'Your camera preview is ready. Start live recognition whenever you are.',
  );
  const [detectedLetter, setDetectedLetter] = useState('');
  const [translation, setTranslation] = useState('');
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState('');

  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const shouldReconnectRef = useRef(false);
  const lastAppendedLetterRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const frameTimerRef = useRef(null);
  const frameInFlightRef = useRef(false);

  function handlePrediction(message) {
    frameInFlightRef.current = false;
    let prediction;

    try {
      prediction = JSON.parse(message);
    } catch (error) {
      console.error('The API sent invalid JSON over the WebSocket.', error);
      setConnectionMessage('The API sent invalid prediction data.');
      return;
    }

    if (
      prediction !== null &&
      typeof prediction === 'object' &&
      (prediction.type === 'error' || prediction.status === 'error')
    ) {
      console.error('The recognition backend reported an error.', prediction);
      setConnectionMessage(
        typeof prediction.message === 'string'
          ? prediction.message
          : 'The backend could not process a camera frame.',
      );
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

    setDetectedLetter((currentLetter) =>
      currentLetter === prediction.letter ? currentLetter : prediction.letter,
    );

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
    setConnectionMessage('Connecting to your live sign stream...');

    let socket;

    try {
      socket = new WebSocket(websocketUrl);
    } catch (error) {
      console.error('Could not create the prediction WebSocket.', error);
      setConnectionMessage('Connection is taking a moment. Trying again...');
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
      setConnectionMessage('Your camera is live and ready to read signs.');
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
      frameInFlightRef.current = false;

      if (shouldReconnectRef.current) {
        setStatus('connecting');
        setConnectionMessage('Connection dropped. Reconnecting to your stream...');
        reconnectTimerRef.current = window.setTimeout(
          connectToWebSocket,
          RECONNECT_DELAY_MS,
        );
      } else {
        setStatus('stopped');
        setConnectionMessage('The live stream is paused.');
      }
    };
  }

  function startRecognition() {
    if (!cameraReady) {
      setConnectionMessage(
        cameraError || 'Allow camera access before starting live recognition.',
      );
      return;
    }

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
    window.clearInterval(frameTimerRef.current);
    frameTimerRef.current = null;
    frameInFlightRef.current = false;

    const socket = socketRef.current;
    socketRef.current = null;
    if (socket) {
      socket.close();
    }

    setStatus('stopped');
    setConnectionMessage('The live stream is paused.');
  }

  function clearTranslation() {
    setDetectedLetter('');
    setTranslation('');
  }

  useEffect(() => {
    let stream;
    let isMounted = true;

    async function startCameraFeed() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraError('This browser does not support webcam access.');
        console.error('This browser does not support webcam access.');
        return;
      }

      let cameraStream;
      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user' },
          audio: false,
        });

        if (!isMounted) {
          cameraStream.getTracks().forEach((track) => track.stop());
          return;
        }

        stream = cameraStream;
        canvasRef.current = document.createElement('canvas');
        const video = videoRef.current;
        if (!video) {
          throw new Error('The camera preview element is unavailable.');
        }

        video.srcObject = cameraStream;
        await video.play();
        setCameraError('');
        setCameraReady(true);

        cameraStream.getVideoTracks().forEach((track) => {
          track.addEventListener('ended', () => {
            if (!isMounted) {
              return;
            }
            setCameraReady(false);
            setCameraError('The camera disconnected. Check it and reload the app.');
            stopRecognition();
          });
        });
      } catch (error) {
        if (!isMounted) {
          cameraStream?.getTracks().forEach((track) => track.stop());
          return;
        }
        cameraStream?.getTracks().forEach((track) => track.stop());
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
        stream = null;
        console.error('Could not start the live camera feed.', error);
        setCameraError(
          error instanceof Error && error.name === 'NotAllowedError'
            ? 'Allow camera access in your browser to use live recognition.'
            : 'Could not start the camera. Check that it is connected and available.',
        );
      }
    }

    startCameraFeed();

    return () => {
      isMounted = false;
      shouldReconnectRef.current = false;
      window.clearTimeout(reconnectTimerRef.current);
      window.clearInterval(frameTimerRef.current);
      frameTimerRef.current = null;
      frameInFlightRef.current = false;

      const socket = socketRef.current;
      socketRef.current = null;
      if (socket && socket.readyState < WebSocket.CLOSING) {
        socket.close();
      }

      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }

      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      canvasRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (status !== 'active' || !cameraReady) {
      window.clearInterval(frameTimerRef.current);
      frameTimerRef.current = null;
      frameInFlightRef.current = false;
      return undefined;
    }

    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) {
      console.error('Could not create the reusable camera frame canvas.');
      setConnectionMessage('Could not capture camera frames for recognition.');
      return undefined;
    }

    frameTimerRef.current = window.setInterval(() => {
      const video = videoRef.current;
      const socket = socketRef.current;
      if (
        !video ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        !video.videoWidth ||
        !video.videoHeight ||
        !socket ||
        socket.readyState !== WebSocket.OPEN ||
        !shouldReconnectRef.current ||
        socket.bufferedAmount > MAX_BUFFERED_BYTES ||
        frameInFlightRef.current
      ) {
        return;
      }

      const scale = Math.min(1, 640 / video.videoWidth);
      const frameWidth = Math.round(video.videoWidth * scale);
      const frameHeight = Math.round(video.videoHeight * scale);
      if (canvas.width !== frameWidth || canvas.height !== frameHeight) {
        canvas.width = frameWidth;
        canvas.height = frameHeight;
      }
      try {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        frameInFlightRef.current = true;

        canvas.toBlob(
          (blob) => {
            if (
              !blob ||
              socketRef.current !== socket ||
              socket.readyState !== WebSocket.OPEN ||
              !shouldReconnectRef.current
            ) {
              frameInFlightRef.current = false;
              return;
            }

            try {
              socket.send(blob);
            } catch (error) {
              frameInFlightRef.current = false;
              console.error('Could not send a camera frame to the backend.', error);
            }
          },
          'image/jpeg',
          0.75,
        );
      } catch (error) {
        frameInFlightRef.current = false;
        console.error('Could not capture a camera frame.', error);
      }
    }, 1000 / FRAME_RATE);

    return () => {
      window.clearInterval(frameTimerRef.current);
      frameTimerRef.current = null;
      frameInFlightRef.current = false;
    };
  }, [cameraReady, status]);

  return (
    <div className="app-shell">
      <Header status={status} />

      <main>
        <div className="intro">
          <div>
            <span className="eyebrow">Your sign assistant</span>
            <h2>Turn hand signs into words.</h2>
            <p>
              Hold your hand in front of the camera and let the app read it in
              real time.
            </p>
          </div>
        </div>

        <div className="workspace">
          <CameraPreview
            active={status === 'active'}
            cameraReady={cameraReady}
            videoRef={videoRef}
          />
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
          {cameraError || connectionMessage}
        </p>
      </main>
    </div>
  );
}

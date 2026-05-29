export class LabApi {
  constructor() {
    this.apiBase = '/api';
    this.wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
    this.httpUrl = `${location.origin}${this.apiBase}`;
    this.socket = null;
    this.connected = false;
    this.reconnectTimer = null;
  }

  async fetchState() {
    const response = await fetch(`${this.apiBase}/state`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`GET /api/state failed: ${response.status}`);
    }
    return response.json();
  }

  async postCommand(payload) {
    const response = await fetch(`${this.apiBase}/command`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`POST /api/command failed: ${response.status} ${text}`);
    }
    return response.json();
  }

  async exportState() {
    const response = await fetch(`${this.apiBase}/export`, { method: 'POST' });
    if (!response.ok) {
      throw new Error(`POST /api/export failed: ${response.status}`);
    }
    return response.json();
  }

  async checkpoint() {
    const response = await fetch(`${this.apiBase}/checkpoint`, { method: 'POST' });
    if (!response.ok) {
      throw new Error(`POST /api/checkpoint failed: ${response.status}`);
    }
    return response.json();
  }

  connect({ onMessage, onStatus }) {
    clearTimeout(this.reconnectTimer);
    onStatus('connecting');
    try {
      this.socket = new WebSocket(this.wsUrl);
    } catch (error) {
      console.warn('WebSocket construction failed', error);
      onStatus('fallback');
      return;
    }

    this.socket.addEventListener('open', () => {
      this.connected = true;
      console.info('Eidos Life Lab WebSocket connected', this.wsUrl);
      onStatus('connected');
    });

    this.socket.addEventListener('message', (event) => {
      try {
        const payload = JSON.parse(event.data);
        onMessage(payload);
      } catch (error) {
        console.warn('Could not parse WebSocket message', error);
      }
    });

    this.socket.addEventListener('close', () => {
      if (this.connected) {
        console.warn('Eidos Life Lab WebSocket closed; HTTP polling fallback will continue.');
      }
      this.connected = false;
      onStatus('fallback');
      this.reconnectTimer = setTimeout(() => this.connect({ onMessage, onStatus }), 2500);
    });

    this.socket.addEventListener('error', (event) => {
      console.warn('Eidos Life Lab WebSocket error; HTTP fallback is active.', event);
      onStatus('fallback');
    });
  }

  async sendCommand(payload) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(payload));
      return { ok: true, via: 'ws' };
    }
    const result = await this.postCommand(payload);
    return { ...result, via: 'http' };
  }
}

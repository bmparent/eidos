export class EidosBackendBridge {
  constructor({ url = '', enabled = false } = {}) { this.url = url; this.enabled = enabled; }
  async predictFrame(frameVector) { if (!this.enabled) return { enabled: false, prediction: null, frameSize: frameVector?.length ?? 0 }; return { enabled: true, prediction: null }; }
  async sendTelemetry(row) { if (!this.enabled) return { enabled: false, accepted: false }; return { enabled: true, accepted: true, row }; }
}

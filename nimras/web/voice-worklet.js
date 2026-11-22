// N.I.M.R.A.S. Voice Engine - microphone capture worklet (runs on the audio thread).
// Downsamples the browser's capture rate (usually 48 kHz) to 16 kHz with a
// box low-pass, and posts 20 ms Int16 frames plus their RMS level.
class NveCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.acc = 0; this.accN = 0; this.pos = 0;
    this.frame = new Int16Array(320);
    this.n = 0; this.sumSq = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i]; this.accN++;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.accN));
        this.acc = 0; this.accN = 0;
        this.frame[this.n++] = v < 0 ? v * 32768 : v * 32767;
        this.sumSq += v * v;
        if (this.n === 320) {
          const out = this.frame.slice(0);
          this.port.postMessage({ pcm: out.buffer, rms: Math.sqrt(this.sumSq / 320) }, [out.buffer]);
          this.n = 0; this.sumSq = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("nve-capture", NveCapture);

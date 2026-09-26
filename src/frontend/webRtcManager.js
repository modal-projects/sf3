const iceServerTimeoutMs = 3000;
const gameIdStorageKey = "sf3-game-id";
const gameIdPattern =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const fallbackIceServers = [{ urls: "stun:stun.l.google.com:19302" }];

export const WebRtcManager = {
  ws: null,
  peer: null,
  dataChannel: null,
  onMessage: null,
  onRemoteStream: null,
  onDisconnect: null,
  gameId: "",
  turnResolver: null,
  hasStarted: false,
  pendingMessages: [],
  pendingRemoteStream: null,
  pendingOutbound: [],
  pendingDisconnect: null,
  pendingIceCandidates: [],
  signalingChain: Promise.resolve(),

  init(callbacks = {}) {
    this.onMessage = callbacks.onMessage || null;
    this.onRemoteStream = callbacks.onRemoteStream || null;
    this.onDisconnect = callbacks.onDisconnect || null;

    if (this.onMessage && this.pendingMessages.length > 0) {
      const messages = [...this.pendingMessages];
      this.pendingMessages = [];
      messages.forEach((raw) => this.onMessage?.(raw));
    }

    if (this.onRemoteStream && this.pendingRemoteStream) {
      this.onRemoteStream(this.pendingRemoteStream);
      this.pendingRemoteStream = null;
    }

    if (this.onDisconnect && this.pendingDisconnect) {
      const message = this.pendingDisconnect;
      this.pendingDisconnect = null;
      this.onDisconnect(message);
      return;
    }

    if (this.hasStarted) {
      return;
    }

    this.hasStarted = true;
    this.connect();
  },

  async connect() {
    try {
      this.gameId = this.getGameId();
      await this.openSignalingSocket();
      const iceServers = await this.getIceServers();
      this.peer = new RTCPeerConnection({ iceServers });
      this.peer.addTransceiver("video", { direction: "recvonly" });

      this.peer.onicecandidate = (event) => {
        if (!event.candidate || !event.candidate.candidate) {
          return;
        }
        this.sendSignal({
          type: "ice_candidate",
          candidate: {
            candidate_sdp: event.candidate.candidate,
            sdpMid: event.candidate.sdpMid,
            sdpMLineIndex: event.candidate.sdpMLineIndex,
          },
        });
      };

      this.peer.ontrack = (event) => {
        const stream =
          event.streams[0] ||
          (event.track ? new MediaStream([event.track]) : null);
        if (!stream) {
          return;
        }
        if (this.onRemoteStream) {
          this.onRemoteStream(stream);
          return;
        }
        this.pendingRemoteStream = stream;
      };

      this.peer.onconnectionstatechange = () => {
        const state = this.peer?.connectionState;
        if (state === "failed" || state === "closed") {
          this.handleDisconnect("Game connection lost.");
        }
      };

      this.dataChannel = this.peer.createDataChannel("game_control");
      this.dataChannel.onopen = () => {
        this.flushPendingOutbound();
      };
      this.dataChannel.onmessage = (event) => {
        const raw = String(event.data);
        if (this.onMessage) {
          this.onMessage(raw);
          return;
        }
        this.pendingMessages.push(raw);
      };
      this.dataChannel.onclose = () => {
        this.handleDisconnect("Game data channel closed.");
      };
      this.flushPendingOutbound();

      const offer = await this.peer.createOffer();
      await this.peer.setLocalDescription(offer);
      this.sendSignal({
        type: offer.type,
        sdp: offer.sdp || "",
      });
    } catch (error) {
      console.error("WebRTC connect error", error);
      this.handleDisconnect("Connection Error");
      this.hasStarted = false;
    }
  },

  async openSignalingSocket() {
    const wsUrl = new URL(`/ws/${this.gameId}`, window.location.href);
    wsUrl.protocol = wsUrl.protocol === "https:" ? "wss:" : "ws:";
    this.ws = new WebSocket(wsUrl.toString());

    this.ws.onmessage = (event) => {
      this.signalingChain = this.signalingChain
        .then(() => this.handleSignalingMessage(String(event.data)))
        .catch((error) => {
          console.error("WebRTC signaling error", error);
          this.handleDisconnect("Connection Error");
        });
    };
    const onSignalingLost = (message) => () => {
      if (!this.peer || this.peer.connectionState !== "connected") {
        this.handleDisconnect(message);
      }
    };
    this.ws.onclose = onSignalingLost("Signaling connection lost.");
    this.ws.onerror = onSignalingLost("Connection Error");

    await new Promise((resolve, reject) => {
      const ws = this.ws;
      if (!ws) {
        reject(new Error("websocket missing"));
        return;
      }

      const onOpen = () => {
        ws.removeEventListener("error", onError);
        ws.removeEventListener("close", onClose);
        resolve();
      };
      const onError = () => {
        ws.removeEventListener("open", onOpen);
        ws.removeEventListener("close", onClose);
        reject(new Error("signaling websocket error"));
      };
      const onClose = () => {
        ws.removeEventListener("open", onOpen);
        ws.removeEventListener("error", onError);
        reject(new Error("signaling websocket closed"));
      };

      ws.addEventListener("open", onOpen, { once: true });
      ws.addEventListener("error", onError, { once: true });
      ws.addEventListener("close", onClose, { once: true });
    });
  },

  async getIceServers() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return fallbackIceServers;
    }

    const iceServerPromise = new Promise((resolve) => {
      this.turnResolver = resolve;
    });
    this.sendSignal({ type: "get_turn_servers" });

    try {
      return await Promise.race([
        iceServerPromise,
        new Promise((resolve) =>
          setTimeout(() => resolve(fallbackIceServers), iceServerTimeoutMs)
        ),
      ]);
    } catch {
      return fallbackIceServers;
    } finally {
      this.turnResolver = null;
    }
  },

  async handleSignalingMessage(raw) {
    let message;
    try {
      message = JSON.parse(raw);
    } catch {
      return;
    }

    if (message.type === "turn_servers" && message.ice_servers && this.turnResolver) {
      this.turnResolver(message.ice_servers);
      return;
    }

    if (message.type === "answer" && this.peer && message.sdp) {
      await this.peer.setRemoteDescription(
        new RTCSessionDescription({ type: "answer", sdp: message.sdp })
      );
      while (this.pendingIceCandidates.length > 0) {
        await this.peer.addIceCandidate(this.pendingIceCandidates.shift());
      }
      return;
    }

    if (message.type === "ice_candidate" && this.peer && message.candidate) {
      const candidate = new RTCIceCandidate({
        candidate: message.candidate.candidate_sdp,
        sdpMid: message.candidate.sdpMid,
        sdpMLineIndex: message.candidate.sdpMLineIndex,
      });
      if (!this.peer.remoteDescription) {
        this.pendingIceCandidates.push(candidate);
        return;
      }
      await this.peer.addIceCandidate(candidate);
    }
  },

  sendSignal(message) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    this.ws.send(JSON.stringify(message));
  },

  send(type, data) {
    const payload = JSON.stringify({ type, data });
    if (!this.dataChannel || this.dataChannel.readyState !== "open") {
      this.pendingOutbound.push(payload);
      return;
    }
    this.dataChannel.send(payload);
  },

  flushPendingOutbound() {
    if (!this.dataChannel || this.dataChannel.readyState !== "open") {
      return;
    }
    while (this.pendingOutbound.length > 0) {
      const payload = this.pendingOutbound.shift();
      if (payload) {
        this.dataChannel.send(payload);
      }
    }
  },

  closeTransports() {
    const dataChannel = this.dataChannel;
    const peer = this.peer;
    const ws = this.ws;
    this.dataChannel = null;
    this.peer = null;
    this.ws = null;

    if (dataChannel) {
      dataChannel.onopen = null;
      dataChannel.onmessage = null;
      dataChannel.onclose = null;
      dataChannel.close();
    }
    if (peer) {
      peer.onicecandidate = null;
      peer.ontrack = null;
      peer.onconnectionstatechange = null;
      peer.close();
    }
    if (ws) {
      ws.onmessage = null;
      ws.onclose = null;
      ws.onerror = null;
      ws.close();
    }
  },

  close() {
    this.onDisconnect = null;
    this.closeTransports();
    this.turnResolver = null;
    this.hasStarted = false;
    this.pendingMessages = [];
    this.pendingRemoteStream = null;
    this.pendingOutbound = [];
    this.pendingDisconnect = null;
    this.pendingIceCandidates = [];
    this.signalingChain = Promise.resolve();
  },

  handleDisconnect(message) {
    this.hasStarted = false;
    this.closeTransports();
    if (this.onDisconnect) {
      this.onDisconnect(message);
      return;
    }
    this.pendingDisconnect = message;
  },

  readStoredGameId() {
    const stored = window.sessionStorage.getItem(gameIdStorageKey);
    if (stored === null) return null;
    if (gameIdPattern.test(stored)) return stored;
    window.sessionStorage.removeItem(gameIdStorageKey);
    return null;
  },

  hasStoredGameId() {
    return this.readStoredGameId() !== null;
  },

  getGameId() {
    const stored = this.readStoredGameId();
    if (stored !== null) return stored;
    const gameId = crypto.randomUUID();
    window.sessionStorage.setItem(gameIdStorageKey, gameId);
    return gameId;
  },

  clearStoredGameId() {
    window.sessionStorage.removeItem(gameIdStorageKey);
  },

  async getLauncherUrl() {
    try {
      const response = await fetch("/api/launcher-url", { cache: "no-store" });
      if (!response.ok) return null;
      const { url } = await response.json();
      return typeof url === "string" && url ? url : null;
    } catch {
      return null;
    }
  },
};

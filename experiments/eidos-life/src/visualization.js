import * as THREE from 'three';

const clamp01 = value => Math.max(0, Math.min(1, value));
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const REGIME_PALETTE = {
  CALIBRATING: {
    background: 0x050812,
    fog: 0x061329,
    key: 0x65d9ff,
    cell: 0x8ee8ff,
    stress: 0xffb86b,
    board: 0x071422,
    height: 0.86,
    fogDensity: 0.034,
    emissive: 0.42,
  },
  GREEN: {
    background: 0x030a0e,
    fog: 0x04221f,
    key: 0x74ffd2,
    cell: 0x9effef,
    stress: 0xffc46e,
    board: 0x06241f,
    height: 1,
    fogDensity: 0.028,
    emissive: 0.58,
  },
  AMBER: {
    background: 0x0d0904,
    fog: 0x2d1a06,
    key: 0xffc15f,
    cell: 0xffe0a3,
    stress: 0xff6f3d,
    board: 0x211306,
    height: 1.12,
    fogDensity: 0.04,
    emissive: 0.72,
  },
  RED: {
    background: 0x100306,
    fog: 0x320710,
    key: 0xff4f6d,
    cell: 0xff9aa8,
    stress: 0xff332f,
    board: 0x24070a,
    height: 0.92,
    fogDensity: 0.055,
    emissive: 0.9,
  },
  BLUE: {
    background: 0x030713,
    fog: 0x061b3d,
    key: 0x73b7ff,
    cell: 0xa9dbff,
    stress: 0x8df7ff,
    board: 0x07172b,
    height: 1.05,
    fogDensity: 0.026,
    emissive: 0.5,
  },
  VIOLET: {
    background: 0x070415,
    fog: 0x20104a,
    key: 0xd191ff,
    cell: 0xf0cbff,
    stress: 0xff5fd7,
    board: 0x130b2d,
    height: 1.22,
    fogDensity: 0.036,
    emissive: 0.84,
  },
};

const REGIME_CLASS = {
  CALIBRATING: 'calibrating',
  GREEN: 'green',
  AMBER: 'amber',
  RED: 'red',
  BLUE: 'blue',
  VIOLET: 'violet',
};

function regimeColor(regime) {
  if (regime === 'RED') return [255, 72, 90];
  if (regime === 'AMBER') return [255, 180, 80];
  if (regime === 'BLUE') return [84, 160, 255];
  if (regime === 'VIOLET') return [210, 130, 255];
  if (regime === 'CALIBRATING') return [120, 140, 165];
  return [80, 255, 198];
}

export class LifeVisualization {
  constructor({ container, engine }) {
    this.engine = engine;
    this.overlays = { surprise: true, memory: true, energy: false, outlines: true, prediction: true };
    this.cellStep = 0.15;
    this.cellSize = 0.108;
    this.boardWidth = engine.width * this.cellStep;
    this.boardDepth = engine.height * this.cellStep;
    this.left = -this.boardWidth / 2;
    this.back = -this.boardDepth / 2;
    this.regime = 'CALIBRATING';
    this.clock = new THREE.Clock();
    this.dragging = false;
    this.cameraYaw = Math.PI * 0.18;
    this.cameraPitch = 0.82;
    this.cameraDistance = 15.8;
    this.pulses = [];

    this.matrix = new THREE.Matrix4();
    this.position = new THREE.Vector3();
    this.scale = new THREE.Vector3(1, 1, 1);
    this.rotation = new THREE.Quaternion();
    this.target = new THREE.Vector3(0, 0.18, 0);
    this.color = new THREE.Color();
    this.stressColor = new THREE.Color();
    this.speciesColor = new THREE.Color();
    this.boardColor = new THREE.Color();

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(REGIME_PALETTE.CALIBRATING.background);
    this.scene.fog = new THREE.FogExp2(REGIME_PALETTE.CALIBRATING.fog, REGIME_PALETTE.CALIBRATING.fogDensity);

    this.world = new THREE.Group();
    this.scene.add(this.world);

    this.camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 120);
    this.updateCamera(0);

    this.createAtmosphere();
    this.createBoard();
    this.createCells();
    this.createOverlays();
    this.createOrganismOutlines();
    this.createCentroidMarkers();
    this.createPredictionSparks();
    this.createPulseMeshes();
    this.resize();
    this.applyRegime('CALIBRATING', { plasticity: 0, surprise: 0 });

    requestAnimationFrame(() => {
      this.createRenderer(container);
      this.bindCameraControls();
      this.resize();
      this.applyRegime('CALIBRATING', { plasticity: 0, surprise: 0 });
    });
  }

  createAtmosphere() {
    this.hemisphere = new THREE.HemisphereLight(0x9cdfff, 0x04070c, 0.72);
    this.scene.add(this.hemisphere);

    this.keyLight = new THREE.DirectionalLight(REGIME_PALETTE.CALIBRATING.key, 2.4);
    this.keyLight.position.set(-3.8, 8, 5.5);
    this.scene.add(this.keyLight);

    this.rimLight = new THREE.PointLight(0x9fe7ff, 3.2, 24, 1.7);
    this.rimLight.position.set(4, 4.8, -5.2);
    this.scene.add(this.rimLight);
  }

  createBoard() {
    this.board = new THREE.Mesh(
      new THREE.PlaneGeometry(this.boardWidth, this.boardDepth),
      new THREE.MeshStandardMaterial({
        color: REGIME_PALETTE.CALIBRATING.board,
        roughness: 0.58,
        metalness: 0.15,
        emissive: 0x04121c,
        emissiveIntensity: 0.32,
        side: THREE.DoubleSide,
      })
    );
    this.board.rotation.x = -Math.PI / 2;
    this.board.position.y = -0.012;
    this.world.add(this.board);

    this.grid = new THREE.GridHelper(this.boardWidth, this.engine.width, 0x226c87, 0x0d2f43);
    this.grid.position.y = 0.006;
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.54;
    this.world.add(this.grid);
  }

  createCells() {
    this.cellMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      vertexColors: true,
      roughness: 0.32,
      metalness: 0.12,
      emissive: REGIME_PALETTE.CALIBRATING.cell,
      emissiveIntensity: REGIME_PALETTE.CALIBRATING.emissive,
    });
    this.cellMesh = new THREE.InstancedMesh(
      new THREE.BoxGeometry(this.cellSize, 1, this.cellSize),
      this.cellMaterial,
      this.engine.size
    );
    this.cellMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.world.add(this.cellMesh);

    this.glowMaterial = new THREE.MeshBasicMaterial({
      color: REGIME_PALETTE.CALIBRATING.cell,
      transparent: true,
      opacity: 0.18,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.glowMesh = new THREE.InstancedMesh(
      new THREE.BoxGeometry(this.cellSize * 1.5, 1, this.cellSize * 1.5),
      this.glowMaterial,
      this.engine.size
    );
    this.glowMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.world.add(this.glowMesh);
  }

  createOverlays() {
    this.fieldTextures = {
      memory: this.createFieldTexture(),
      energy: this.createFieldTexture(),
      surprise: this.createFieldTexture(),
      prediction: this.createFieldTexture(),
      regime: this.createFieldTexture(),
    };

    this.memoryOverlay = this.createOverlayPlane(this.fieldTextures.memory.texture, 0.026, 0.62);
    this.energyOverlay = this.createOverlayPlane(this.fieldTextures.energy.texture, 0.038, 0.5);
    this.surpriseOverlay = this.createOverlayPlane(this.fieldTextures.surprise.texture, 0.052, 0.58);
    this.predictionOverlay = this.createOverlayPlane(this.fieldTextures.prediction.texture, 0.066, 0.54);
    this.regimeOverlay = this.createOverlayPlane(this.fieldTextures.regime.texture, 0.018, 0.26);
    this.world.add(this.regimeOverlay, this.memoryOverlay, this.energyOverlay, this.surpriseOverlay, this.predictionOverlay);
  }

  createFieldTexture() {
    const data = new Uint8Array(this.engine.size * 4);
    const texture = new THREE.DataTexture(data, this.engine.width, this.engine.height, THREE.RGBAFormat);
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.NearestFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.needsUpdate = true;
    return { data, texture };
  }

  createOverlayPlane(texture, y, opacity) {
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(this.boardWidth, this.boardDepth),
      new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.DoubleSide,
      })
    );
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.y = y;
    return mesh;
  }

  createOrganismOutlines() {
    this.maxOrganismOutlines = 256;
    this.organismSegments = 12;
    const vertexCount = this.maxOrganismOutlines * this.organismSegments * 2;
    this.organismPositions = new Float32Array(vertexCount * 3);
    this.organismColors = new Float32Array(vertexCount * 3);
    this.organismGeometry = new THREE.BufferGeometry();
    this.organismGeometry.setAttribute('position', new THREE.BufferAttribute(this.organismPositions, 3));
    this.organismGeometry.setAttribute('color', new THREE.BufferAttribute(this.organismColors, 3));
    this.organismGeometry.setDrawRange(0, 0);
    this.organismLines = new THREE.LineSegments(
      this.organismGeometry,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.9,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    this.world.add(this.organismLines);
  }

  createCentroidMarkers() {
    this.maxCentroids = 128;
    this.centroidMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.centroidMesh = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.055, 10, 8),
      this.centroidMaterial,
      this.maxCentroids
    );
    this.centroidMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.world.add(this.centroidMesh);
  }

  createPredictionSparks() {
    this.maxSparks = 80;
    this.sparkMaterial = new THREE.MeshBasicMaterial({
      color: 0xff58e8,
      transparent: true,
      opacity: 0.88,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.sparkMesh = new THREE.InstancedMesh(
      new THREE.IcosahedronGeometry(0.045, 1),
      this.sparkMaterial,
      this.maxSparks
    );
    this.sparkMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.world.add(this.sparkMesh);
  }

  createPulseMeshes() {
    const geometry = new THREE.RingGeometry(0.98, 1.04, 96);
    for (let i = 0; i < 5; i++) {
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshBasicMaterial({
          color: 0x9efcff,
          transparent: true,
          opacity: 0,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
          side: THREE.DoubleSide,
        })
      );
      mesh.rotation.x = -Math.PI / 2;
      mesh.visible = false;
      this.world.add(mesh);
      this.pulses.push({ mesh, active: false, age: 0, power: 0.8 });
    }
  }

  createRenderer(container) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setClearColor(REGIME_PALETTE.CALIBRATING.background, 1);
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    this.renderer.domElement.style.display = 'block';
    this.renderer.domElement.style.position = 'fixed';
    this.renderer.domElement.style.inset = '0';
    this.renderer.domElement.style.zIndex = '0';
    container.appendChild(this.renderer.domElement);
  }

  bindCameraControls() {
    const canvas = this.renderer.domElement;
    canvas.addEventListener('pointerdown', event => {
      this.dragging = true;
      this.lastPointerX = event.clientX;
      this.lastPointerY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
      if (!this.dragging) return;
      const dx = event.clientX - this.lastPointerX;
      const dy = event.clientY - this.lastPointerY;
      this.lastPointerX = event.clientX;
      this.lastPointerY = event.clientY;
      this.cameraYaw -= dx * 0.006;
      this.cameraPitch = clamp(this.cameraPitch + dy * 0.004, 0.42, 1.18);
      this.updateCamera(0);
    });
    canvas.addEventListener('pointerup', event => {
      this.dragging = false;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointercancel', () => {
      this.dragging = false;
    });
    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      this.cameraDistance = clamp(this.cameraDistance + event.deltaY * 0.01, 10.8, 23);
      this.updateCamera(0);
    }, { passive: false });
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    if (this.renderer) this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  reset() {
    this.world.rotation.y = 0;
    for (const pulse of this.pulses) {
      pulse.active = false;
      pulse.mesh.visible = false;
    }
  }

  pulse({ x = this.engine.width / 2, y = this.engine.height / 2, power = 0.8 } = {}) {
    const pulse = this.pulses.find(item => !item.active) || this.pulses[0];
    pulse.active = true;
    pulse.age = 0;
    pulse.power = power;
    pulse.mesh.visible = true;
    pulse.mesh.position.set(
      this.left + x * this.cellStep + this.cellStep / 2,
      0.09,
      this.back + y * this.cellStep + this.cellStep / 2
    );
    pulse.mesh.scale.setScalar(0.18);
    pulse.mesh.material.opacity = 0.72;
  }

  updateCamera(delta) {
    if (!this.dragging) this.cameraYaw += delta * 0.045;
    const horizontal = Math.cos(this.cameraPitch) * this.cameraDistance;
    this.camera.position.set(
      Math.sin(this.cameraYaw) * horizontal,
      Math.sin(this.cameraPitch) * this.cameraDistance,
      Math.cos(this.cameraYaw) * horizontal
    );
    this.camera.lookAt(this.target);
  }

  applyRegime(regime, metrics) {
    const palette = REGIME_PALETTE[regime] || REGIME_PALETTE.GREEN;
    this.regime = regime;
    this.scene.background.set(palette.background);
    this.scene.fog.color.set(palette.fog);
    this.scene.fog.density = palette.fogDensity + (metrics.surprise || 0) * 0.025;
    if (this.renderer) this.renderer.setClearColor(palette.background, 1);
    this.keyLight.color.set(palette.key);
    this.keyLight.intensity = 2.1 + (metrics.plasticity || 0) * 1.6;
    this.rimLight.color.set(palette.cell);
    this.rimLight.intensity = 2.4 + (metrics.surprise || 0) * 5.5;
    this.cellMaterial.emissive.set(palette.cell);
    this.cellMaterial.emissiveIntensity = palette.emissive + (metrics.plasticity || 0) * 0.45;
    this.glowMaterial.color.set(palette.cell);
    this.board.material.color.set(palette.board);
  }

  render({ metrics, organisms = [], prediction = null, localRegimes = null, genomeRegistry = null, selectedOrganism = null }) {
    if (!this.renderer) return;

    const delta = Math.min(this.clock.getDelta(), 0.05);
    this.updateCamera(delta);
    this.applyRegime(metrics.regime, metrics);

    const aliveCount = this.updateLiveCells(metrics, genomeRegistry);
    this.updateFieldOverlays(metrics, prediction, localRegimes);
    this.updateOrganismOutlines(organisms, metrics, selectedOrganism);
    this.updateCentroidMarkers(organisms, selectedOrganism);
    this.updatePredictionSparks(prediction);
    this.updatePulses(delta, metrics);

    this.cellMesh.count = aliveCount;
    this.glowMesh.count = aliveCount;
    this.centroidMesh.instanceMatrix.needsUpdate = true;
    this.sparkMesh.instanceMatrix.needsUpdate = true;
    this.cellMesh.instanceMatrix.needsUpdate = true;
    this.glowMesh.instanceMatrix.needsUpdate = true;
    if (this.cellMesh.instanceColor) this.cellMesh.instanceColor.needsUpdate = true;

    this.renderer.render(this.scene, this.camera);
  }

  updateLiveCells(metrics, genomeRegistry = null) {
    const { alive, age, energy, species, genomeId, memoryField, stress, width, size } = this.engine;
    const palette = REGIME_PALETTE[metrics.regime] || REGIME_PALETTE.GREEN;
    const regimeHeight = palette.height;
    const shimmer = metrics.regime === 'VIOLET' ? Math.sin(performance.now() * 0.008) * 0.16 : 0;
    let cursor = 0;

    for (let i = 0; i < size; i++) {
      if (!alive[i]) continue;
      const x = i % width;
      const y = Math.floor(i / width);
      const ageFactor = clamp01(age[i] / 80);
      const energyFactor = clamp01(energy[i]);
      const stressFactor = clamp01(stress[i]);
      const memoryFactor = clamp01(memoryField[i]);
      const height = (
        0.08 +
        ageFactor * 0.72 +
        energyFactor * 0.42 +
        stressFactor * 0.42 +
        metrics.plasticity * 0.34 +
        metrics.surprise * 0.8 +
        shimmer * memoryFactor
      ) * regimeHeight;
      const px = this.left + x * this.cellStep + this.cellStep / 2;
      const pz = this.back + y * this.cellStep + this.cellStep / 2;

      this.position.set(px, height / 2, pz);
      this.scale.set(1, height, 1);
      this.matrix.compose(this.position, this.rotation, this.scale);
      this.cellMesh.setMatrixAt(cursor, this.matrix);

      const genome = genomeRegistry?.get?.(genomeId[i]);
      const speciesHue = genome?.traits.colorHue ?? (((species[i] || 1) * 0.08 + 0.48) % 1);
      const light = clamp(0.48 + energyFactor * 0.25 + ageFactor * 0.14, 0.38, 0.82);
      this.speciesColor.setHSL(speciesHue, 0.82, light);
      this.color.set(palette.cell).lerp(this.speciesColor, 0.44);
      this.stressColor.set(palette.stress);
      this.color.lerp(this.stressColor, stressFactor * 0.72 + metrics.surprise * 0.28);
      this.cellMesh.setColorAt(cursor, this.color);

      const glowScale = 1.28 + energyFactor * 0.34 + metrics.plasticity * 0.2;
      this.position.set(px, height / 2, pz);
      this.scale.set(glowScale, height * 1.08, glowScale);
      this.matrix.compose(this.position, this.rotation, this.scale);
      this.glowMesh.setMatrixAt(cursor, this.matrix);
      cursor++;
    }

    this.glowMaterial.opacity = clamp(0.12 + metrics.plasticity * 0.2 + metrics.surprise * 0.22, 0.1, 0.38);
    return cursor;
  }

  updateFieldOverlays(metrics, prediction = null, localRegimes = null) {
    const { energy, nutrientField, wasteField, memoryField, anomalyField, signalField, stress, width, height } = this.engine;
    this.memoryOverlay.visible = this.overlays.memory;
    this.energyOverlay.visible = this.overlays.energy;
    this.surpriseOverlay.visible = this.overlays.surprise;
    this.predictionOverlay.visible = this.overlays.prediction;
    this.regimeOverlay.visible = this.overlays.surprise || this.overlays.memory;
    this.grid.visible = this.overlays.outlines;

    const memory = this.fieldTextures.memory.data;
    const energyData = this.fieldTextures.energy.data;
    const surprise = this.fieldTextures.surprise.data;
    const predictionData = this.fieldTextures.prediction.data;
    const regimeData = this.fieldTextures.regime.data;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const source = y * width + x;
        const target = ((height - 1 - y) * width + x) * 4;
        const m = clamp01(memoryField[source]);
        const e = clamp01((energy[source] + (nutrientField?.[source] || 0)) * 0.5);
        const waste = clamp01(wasteField?.[source] || 0);
        const s = clamp01((anomalyField[source] || 0) * 1.6 + stress[source] * 0.9 + waste * 0.9 + metrics.surprise * 0.8);
        const sig = clamp01(signalField[source] || 0);
        const predicted = prediction?.predicted?.[source] || 0;
        const error = clamp01(prediction?.errorField?.[source] || 0);

        memory[target] = 96 + m * 120;
        memory[target + 1] = 68 + sig * 80;
        memory[target + 2] = 255;
        memory[target + 3] = m * 180;

        energyData[target] = 36;
        energyData[target + 1] = 180 + e * 75;
        energyData[target + 2] = 180 + sig * 60;
        energyData[target + 3] = e * 130;

        surprise[target] = 255;
        surprise[target + 1] = 56 + s * 120;
        surprise[target + 2] = metrics.regime === 'VIOLET' ? 255 : 72;
        surprise[target + 3] = s * 190;

        predictionData[target] = error ? 255 : 70;
        predictionData[target + 1] = predicted ? 210 : 40;
        predictionData[target + 2] = predicted ? 255 : 180;
        predictionData[target + 3] = clamp01(predicted * 0.38 + error * 0.9) * 170;

        const localRegime = localRegimes?.regimeAtCell?.(x, y) || metrics.regime;
        const [rr, rg, rb] = regimeColor(localRegime);
        regimeData[target] = rr;
        regimeData[target + 1] = rg;
        regimeData[target + 2] = rb;
        regimeData[target + 3] = localRegime === 'GREEN' ? 24 : 70;
      }
    }

    this.fieldTextures.memory.texture.needsUpdate = this.overlays.memory;
    this.fieldTextures.energy.texture.needsUpdate = this.overlays.energy;
    this.fieldTextures.surprise.texture.needsUpdate = this.overlays.surprise;
    this.fieldTextures.prediction.texture.needsUpdate = this.overlays.prediction;
    this.fieldTextures.regime.texture.needsUpdate = this.regimeOverlay.visible;
  }

  updateOrganismOutlines(organisms, metrics, selectedOrganism = null) {
    this.organismLines.visible = this.overlays.outlines;
    if (!this.overlays.outlines) return;

    let largestMass = 1;
    const limit = Math.min(organisms.length, this.maxOrganismOutlines);
    for (let i = 0; i < limit; i++) largestMass = Math.max(largestMass, organisms[i].mass || 1);
    let vertex = 0;

    for (let i = 0; i < limit; i++) {
      const organism = organisms[i];
      const box = organism.boundingBox;
      const minX = this.left + box.minX * this.cellStep;
      const maxX = this.left + (box.maxX + 1) * this.cellStep;
      const minZ = this.back + box.minY * this.cellStep;
      const maxZ = this.back + (box.maxY + 1) * this.cellStep;
      const lift = 0.08 + clamp01(organism.mass / largestMass) * 0.28;
      const top = lift + 0.18 + clamp01(organism.threatScore || 0) * 0.42;
      const threat = clamp01(organism.threatScore || organism.meanStress || 0);
      const novelty = clamp01(organism.noveltyScore || 0);
      this.color.set(metrics.regime === 'RED' ? 0xff6a4d : 0x7df3ff).lerp(this.stressColor.set(0xff4b69), threat);
      this.color.lerp(this.speciesColor.set(0xd694ff), novelty * 0.55);
      if (selectedOrganism?.id === organism.id) this.color.set(0xffffff);

      vertex = this.addBoxLines(vertex, minX, maxX, minZ, maxZ, lift, top, this.color);
    }

    this.organismGeometry.setDrawRange(0, vertex);
    this.organismGeometry.attributes.position.needsUpdate = true;
    this.organismGeometry.attributes.color.needsUpdate = true;
  }

  updateCentroidMarkers(organisms, selectedOrganism = null) {
    const limit = Math.min(organisms.length, this.maxCentroids);
    this.centroidMesh.count = limit;
    for (let i = 0; i < limit; i++) {
      const organism = organisms[i];
      const px = this.left + organism.centroid.x * this.cellStep + this.cellStep / 2;
      const pz = this.back + organism.centroid.y * this.cellStep + this.cellStep / 2;
      const scale = selectedOrganism?.id === organism.id ? 2.4 : 1 + clamp01(organism.fitnessScore || 0) * 1.2;
      this.position.set(px, 0.34 + clamp01(organism.mass / 80) * 0.7, pz);
      this.scale.setScalar(scale);
      this.matrix.compose(this.position, this.rotation, this.scale);
      this.centroidMesh.setMatrixAt(i, this.matrix);
    }
  }

  updatePredictionSparks(prediction = null) {
    const sparks = this.overlays.prediction ? (prediction?.sparks || []) : [];
    const limit = Math.min(sparks.length, this.maxSparks);
    this.sparkMesh.count = limit;
    for (let i = 0; i < limit; i++) {
      const spark = sparks[i];
      const px = this.left + spark.x * this.cellStep + this.cellStep / 2;
      const pz = this.back + spark.y * this.cellStep + this.cellStep / 2;
      const scale = 0.8 + (i % 5) * 0.16;
      this.position.set(px, 0.55 + (i % 3) * 0.04, pz);
      this.scale.setScalar(scale);
      this.matrix.compose(this.position, this.rotation, this.scale);
      this.sparkMesh.setMatrixAt(i, this.matrix);
    }
  }

  addBoxLines(vertex, minX, maxX, minZ, maxZ, bottom, top, color) {
    const corners = [
      [minX, bottom, minZ], [maxX, bottom, minZ], [maxX, bottom, maxZ], [minX, bottom, maxZ],
      [minX, top, minZ], [maxX, top, minZ], [maxX, top, maxZ], [minX, top, maxZ],
    ];
    const segments = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];

    for (const [a, b] of segments) {
      vertex = this.writeLineVertex(vertex, corners[a], color);
      vertex = this.writeLineVertex(vertex, corners[b], color);
    }
    return vertex;
  }

  writeLineVertex(vertex, point, color) {
    const p = vertex * 3;
    this.organismPositions[p] = point[0];
    this.organismPositions[p + 1] = point[1];
    this.organismPositions[p + 2] = point[2];
    this.organismColors[p] = color.r;
    this.organismColors[p + 1] = color.g;
    this.organismColors[p + 2] = color.b;
    return vertex + 1;
  }

  updatePulses(delta, metrics) {
    const palette = REGIME_PALETTE[metrics.regime] || REGIME_PALETTE.GREEN;
    for (const pulse of this.pulses) {
      if (!pulse.active) continue;
      pulse.age += delta;
      const t = pulse.age / 1.45;
      if (t >= 1) {
        pulse.active = false;
        pulse.mesh.visible = false;
        continue;
      }
      const radius = 0.25 + t * 6.8 * pulse.power;
      const fade = (1 - t) * (0.35 + pulse.power * 0.45);
      pulse.mesh.scale.setScalar(radius);
      pulse.mesh.material.opacity = fade;
      pulse.mesh.material.color.set(metrics.regime === 'RED' ? palette.stress : palette.key);
      pulse.mesh.position.y = 0.1 + Math.sin(t * Math.PI) * 0.18;
    }
  }
}

export { REGIME_CLASS };

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const FIELD_KEYS = {
  energy: 'energy',
  memory: 'memory',
  signal: 'signal',
  nutrient: 'nutrient',
  waste: 'waste',
  stress: 'stress'
};

export class LabRenderer {
  constructor(mount, onCellClick) {
    this.mount = mount;
    this.onCellClick = onCellClick;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x06100f);
    this.camera = new THREE.OrthographicCamera(-45, 45, 45, -45, 0.1, 500);
    this.camera.position.set(0, 86, 0.001);
    this.camera.lookAt(0, 0, 0);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    this.renderer.setSize(mount.clientWidth, mount.clientHeight);
    this.mount.appendChild(this.renderer.domElement);
    this.camera.up.set(0, 0, -1);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enableRotate = false;
    this.controls.screenSpacePanning = true;
    this.controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN
    };

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.dummy = new THREE.Object3D();
    this.color = new THREE.Color();
    this.mesh = null;
    this.snapshot = null;
    this.fieldView = 'lineage';
    this.quality = 'medium';
    this.pointerDown = null;

    this.addLights();
    this.addFloor();
    this.bindEvents();
    this.resize();
    this.animate();
  }

  addLights() {
    const hemi = new THREE.HemisphereLight(0xcbe6dc, 0x08100d, 2.4);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xf3e3b2, 2.1);
    key.position.set(-18, 42, 28);
    this.scene.add(key);
  }

  addFloor() {
    const grid = new THREE.GridHelper(72, 72, 0x24514a, 0x10211f);
    grid.position.y = -0.02;
    this.scene.add(grid);
  }

  bindEvents() {
    window.addEventListener('resize', () => this.resize());
    const canvas = this.renderer.domElement;
    canvas.addEventListener('pointerdown', (event) => {
      this.pointerDown = { x: event.clientX, y: event.clientY, button: event.button };
    });
    canvas.addEventListener('pointerup', (event) => {
      if (!this.pointerDown || this.pointerDown.button !== 0) {
        return;
      }
      const distance = Math.hypot(event.clientX - this.pointerDown.x, event.clientY - this.pointerDown.y);
      this.pointerDown = null;
      if (distance > 5) {
        return;
      }
      const cell = this.pickCell(event);
      if (cell) {
        this.onCellClick(cell);
      }
    });
    canvas.addEventListener('contextmenu', (event) => event.preventDefault());
  }

  resize() {
    const width = Math.max(320, this.mount.clientWidth);
    const height = Math.max(320, this.mount.clientHeight);
    this.renderer.setSize(width, height);
    const aspect = width / height;
    const board = this.snapshot ? Math.max(this.snapshot.width, this.snapshot.height) : 72;
    const frustum = board * 0.64;
    this.camera.left = -frustum * aspect;
    this.camera.right = frustum * aspect;
    this.camera.top = frustum;
    this.camera.bottom = -frustum;
    this.camera.updateProjectionMatrix();
  }

  setFieldView(fieldView) {
    this.fieldView = fieldView;
    if (this.snapshot) {
      this.updateSnapshot(this.snapshot, { refreshOnly: true });
    }
  }

  setQuality(quality) {
    this.quality = quality;
    if (quality === 'low') {
      this.renderer.setPixelRatio(1);
    } else if (quality === 'high') {
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    } else {
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    }
    if (this.snapshot) {
      this.updateSnapshot(this.snapshot, { refreshOnly: true });
    }
  }

  topDown() {
    this.controls.enableRotate = false;
    this.camera.up.set(0, 0, -1);
    this.camera.position.set(0, 86, 0);
    this.camera.lookAt(0, 0, 0);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  tiltView() {
    this.controls.enableRotate = true;
    this.camera.up.set(0, 1, 0);
    this.camera.position.set(24, 52, 58);
    this.camera.lookAt(0, 0, 0);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  fitBoard() {
    this.controls.target.set(0, 0, 0);
    this.resize();
    this.controls.update();
  }

  ensureMesh(snapshot) {
    const count = snapshot.width * snapshot.height;
    if (this.mesh && this.mesh.count === count) {
      return;
    }
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
    }
    const geometry = new THREE.BoxGeometry(0.88, 1, 0.88);
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.74,
      metalness: 0.06
    });
    this.mesh = new THREE.InstancedMesh(geometry, material, count);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
    this.scene.add(this.mesh);
  }

  updateSnapshot(snapshot) {
    this.snapshot = snapshot;
    this.ensureMesh(snapshot);
    const width = snapshot.width;
    const height = snapshot.height;
    const alive = snapshot.alive;
    const energy = snapshot.energy;
    const lineage = snapshot.lineage;
    const qualityDeadHeight = this.quality === 'low' ? 0.025 : 0.045;
    for (let index = 0; index < alive.length; index += 1) {
      const x = index % width;
      const y = Math.floor(index / width);
      const isAlive = alive[index] === 1;
      const heightScale = isAlive ? 0.36 + energy[index] * 1.15 : qualityDeadHeight;
      this.dummy.position.set(x - width / 2 + 0.5, heightScale / 2, y - height / 2 + 0.5);
      this.dummy.scale.set(1, heightScale, 1);
      this.dummy.updateMatrix();
      this.mesh.setMatrixAt(index, this.dummy.matrix);
      this.colorForCell(snapshot, index, isAlive, lineage[index]);
      this.mesh.setColorAt(index, this.color);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) {
      this.mesh.instanceColor.needsUpdate = true;
    }
  }

  colorForCell(snapshot, index, isAlive, lineageId) {
    if (this.fieldView === 'alive') {
      this.color.set(isAlive ? 0x9ee7b4 : 0x10211e);
      return;
    }
    if (this.fieldView === 'lineage') {
      if (!isAlive) {
        const memory = snapshot.memoryField[index] || 0;
        this.color.setRGB(0.05 + memory * 0.08, 0.10 + memory * 0.14, 0.09 + memory * 0.10);
        return;
      }
      this.color.setHSL(((lineageId * 0.61803398875) % 1), 0.58, 0.56);
      return;
    }
    const key = FIELD_KEYS[this.fieldView] || 'energy';
    const value = snapshot[key][index] || 0;
    const aliveBoost = isAlive ? 0.14 : 0;
    if (key === 'waste') {
      this.color.setRGB(0.12 + value * 0.58 + aliveBoost, 0.08 + value * 0.18, 0.06 + value * 0.08);
    } else if (key === 'stress') {
      this.color.setRGB(0.11 + value * 0.70 + aliveBoost, 0.09 + value * 0.34, 0.08 + value * 0.10);
    } else if (key === 'nutrient') {
      this.color.setRGB(0.06 + value * 0.22, 0.14 + value * 0.62 + aliveBoost, 0.10 + value * 0.20);
    } else if (key === 'signal') {
      this.color.setRGB(0.07 + value * 0.22, 0.12 + value * 0.42 + aliveBoost, 0.13 + value * 0.62);
    } else if (key === 'memory') {
      this.color.setRGB(0.10 + value * 0.42 + aliveBoost, 0.11 + value * 0.32, 0.15 + value * 0.50);
    } else {
      this.color.setRGB(0.07 + value * 0.55 + aliveBoost, 0.13 + value * 0.58, 0.11 + value * 0.24);
    }
  }

  pickCell(event) {
    if (!this.mesh || !this.snapshot) {
      return null;
    }
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObject(this.mesh);
    if (!hits.length || hits[0].instanceId == null) {
      return null;
    }
    const index = hits[0].instanceId;
    return {
      index,
      x: index % this.snapshot.width,
      y: Math.floor(index / this.snapshot.width)
    };
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

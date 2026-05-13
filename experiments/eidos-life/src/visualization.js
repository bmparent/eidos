import * as THREE from 'three';
export class LifeVisualization {
  constructor({container,engine}){
    this.engine=engine; this.overlays={surprise:true,memory:true,energy:false,outlines:true};
    this.scene=new THREE.Scene(); this.camera=new THREE.PerspectiveCamera(48,innerWidth/innerHeight,0.1,100); this.camera.position.set(0,10,14);
    this.renderer=new THREE.WebGLRenderer({antialias:true}); this.renderer.setSize(innerWidth,innerHeight); container.appendChild(this.renderer.domElement);
    const geo=new THREE.PlaneGeometry(engine.width*0.12,engine.height*0.12,engine.width-1,engine.height-1);
    const mat=new THREE.MeshBasicMaterial({vertexColors:true,wireframe:false,side:THREE.DoubleSide});
    this.mesh=new THREE.Mesh(geo,mat); this.mesh.rotation.x=-Math.PI/2; this.scene.add(this.mesh); this.colors=new Float32Array(geo.attributes.position.count*3); geo.setAttribute('color',new THREE.BufferAttribute(this.colors,3));
    addEventListener('resize',()=>{this.camera.aspect=innerWidth/innerHeight;this.camera.updateProjectionMatrix();this.renderer.setSize(innerWidth,innerHeight);});
  }
  render({metrics}){
    const {alive,energy,memoryField,stress,size}=this.engine;
    for(let i=0;i<size;i++){
      const p=i*3; const life=alive[i]; const e=energy[i]; const m=memoryField[i]; const s=stress[i];
      this.colors[p]=life?0.2+s*0.8:m*0.7; this.colors[p+1]=life?0.6+e*0.3:e*0.5; this.colors[p+2]=life?0.9-m*0.3:0.2+s*0.6;
      if(this.overlays.energy){ this.colors[p+1]=Math.max(this.colors[p+1],e); }
      if(this.overlays.surprise){ this.colors[p]=Math.max(this.colors[p],metrics.surprise*0.8); }
    }
    this.mesh.geometry.attributes.color.needsUpdate=true;
    this.renderer.render(this.scene,this.camera);
  }
}
